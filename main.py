#!/usr/bin/env python3
"""
Senior Health Dialog Pipeline
1. Fetch sensor data from API
2. Detect anomalies
3. RAG retrieval for context
4. Gemma4 generates questions based on anomalies
5. Piper TTS speaks questions to senior
6. Whisper STT converts senior's voice answer to text
7. Log full dialog to dialog.log
"""

import os
import sys
import json
import wave
import tempfile
import subprocess
import datetime
import logging
import requests
import numpy as np
import scipy.signal as signal
import pyaudio
import whisper
import torch

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
import glob

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

# API
API_URL  = "http://relay.emtake.com/api/query"
ACCOUNT  = "test10@test.com"
DATE     = datetime.datetime.now().strftime("%Y-%m-%d")
PREV_DATE = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

# Audio MIC
MIC_CHANNELS   = 2
MIC_RATE       = 44100
MIC_FORMAT     = pyaudio.paInt16
MIC_CHUNK      = 1024
TARGET_RATE    = 16000
SILENCE_THRESHOLD  = 500
SILENCE_DURATION   = 2.0
MAX_RECORD_SEC     = 30

# Audio Speaker
SPK_DEVICE     = "plughw:0,0"

# Piper TTS
PIPER_BIN      = "piper"
PIPER_MODEL    = "/home/soo/Work/mom-i/senior/piper/en_US-amy-medium.onnx"

# Whisper
WHISPER_MODEL  = "base"

# RAG
RAG_DIR        = "/home/soo/Work/mom-i/senior/rag"
CHROMA_DIR     = "/home/soo/Work/mom-i/senior/chroma_db"
EMBED_MODEL    = "nomic-embed-text"
LLM_MODEL      = "gemma4:e2b"
OLLAMA_URL     = "http://localhost:11434"

# Log
LOG_FILE       = "/home/soo/Work/mom-i/senior/dialog.log"

# Anomaly thresholds
THRESHOLDS = {
    "wake_up":        3,      # times
    "sleep_duration": 360,    # minutes (6 hours)
    "breath_max":     20,     # breaths/min
    "breath_min":     8,
    "indoor_temp_max":28.0,   # celsius
    "indoor_temp_min":16.0,
    "db_max":         60,     # decibels
    "body_temp_max":  37.5,   # celsius
    "body_temp_min":  35.0,
}


# ─────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────
def setup_logger():
    logger = logging.getLogger("dialog")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger = setup_logger()


# ─────────────────────────────────────────
# STEP 1: FETCH SENSOR DATA
# ─────────────────────────────────────────
def fetch_sensor(cmd):
    payload = {
        "Type":    "LLMREPORT",
        "Account": ACCOUNT,
        "CMD":     cmd,
        "val":     "1",
        "date":    DATE
    }
    try:
        resp = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=10
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"Sensor fetch failed [{cmd}]: {e}")
        return None


def parse_response(raw):
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except:
            return {}
    return raw or {}


def fetch_all_sensors():
    logger.info("Fetching all sensor data...")
    return {
        "sleep":       fetch_sensor("SleepData"),
        "temp":        fetch_sensor("Temp"),
        "breath":      fetch_sensor("Breath"),
        "db":          fetch_sensor("dB"),
        "indoor_temp": fetch_sensor("IndoorTemp"),
    }


# ─────────────────────────────────────────
# STEP 2: DETECT ANOMALIES
# ─────────────────────────────────────────
def get_test_data(parsed, date):
    """Get data for TEST10 for given date, fallback to PREV_DATE."""
    test_key = next((k for k in parsed if k.startswith("TEST")), None)
    if not test_key:
        return {}
    day_data = parsed[test_key].get(date) or parsed[test_key].get(PREV_DATE, {})
    return day_data


def detect_anomalies(sensor_data):
    anomalies = []

    # Sleep
    sleep_parsed = parse_response(sensor_data.get("sleep", {}))
    test_key = next((k for k in sleep_parsed if k.startswith("TEST")), None)
    if test_key:
        day_data     = sleep_parsed[test_key].get(DATE) or sleep_parsed[test_key].get(PREV_DATE, {})
        sessions     = day_data.get("sessions", [])
        total_wakeup = sum(s.get("wake_up", 0) for s in sessions)
        total_dur    = sum(s.get("duration_min", 0) for s in sessions)

        logger.info(f"Sleep: sessions={len(sessions)}, wakeups={total_wakeup}, duration={total_dur}min")

        if total_wakeup > THRESHOLDS["wake_up"]:
            anomalies.append({
                "type":    "frequent_waking",
                "value":   total_wakeup,
                "message": f"Woke up {total_wakeup} times during sleep (threshold: >{THRESHOLDS['wake_up']})"
            })
        if 0 < total_dur < THRESHOLDS["sleep_duration"]:
            anomalies.append({
                "type":    "short_sleep",
                "value":   total_dur,
                "message": f"Short sleep duration: {total_dur} minutes (threshold: <{THRESHOLDS['sleep_duration']})"
            })

    # Breathing
    breath_parsed = parse_response(sensor_data.get("breath", {}))
    day_data = get_test_data(breath_parsed, DATE)
    if day_data:
        b_min = day_data.get("Min", 0)
        b_max = day_data.get("Max", 0)
        logger.info(f"Breath: min={b_min}, max={b_max}")
        if b_max > THRESHOLDS["breath_max"] or b_min < THRESHOLDS["breath_min"]:
            anomalies.append({
                "type":    "abnormal_breathing",
                "value":   f"{b_min}-{b_max}",
                "message": f"Abnormal breathing rate: {b_min}-{b_max} breaths/min"
            })

    # Indoor Temperature
    indoor_parsed = parse_response(sensor_data.get("indoor_temp", {}))
    day_data = get_test_data(indoor_parsed, DATE)
    if day_data:
        t_min = day_data.get("Min", 0)
        t_max = day_data.get("Max", 0)
        logger.info(f"IndoorTemp: min={t_min}, max={t_max}")
        if t_max > THRESHOLDS["indoor_temp_max"] or t_min < THRESHOLDS["indoor_temp_min"]:
            anomalies.append({
                "type":    "abnormal_room_temp",
                "value":   f"{t_min}-{t_max}",
                "message": f"Room temperature out of range: {t_min}-{t_max}°C"
            })

    # Noise
    db_parsed = parse_response(sensor_data.get("db", {}))
    day_data = get_test_data(db_parsed, DATE)
    if day_data:
        db_max = day_data.get("Max", 0)
        logger.info(f"dB: max={db_max}")
        if db_max > THRESHOLDS["db_max"]:
            anomalies.append({
                "type":    "high_noise",
                "value":   db_max,
                "message": f"High noise level during sleep: {db_max}dB"
            })

    # Body Temperature (skip if sensor error: values below 10)
    temp_parsed = parse_response(sensor_data.get("temp", {}))
    day_data = get_test_data(temp_parsed, DATE)
    if day_data:
        bt_min = day_data.get("Min", 0)
        bt_max = day_data.get("Max", 0)
        logger.info(f"BodyTemp: min={bt_min}, max={bt_max}")
        if bt_max > 10:   # skip if sensor error
            if bt_max > THRESHOLDS["body_temp_max"]:
                anomalies.append({
                    "type":    "high_body_temp",
                    "value":   bt_max,
                    "message": f"High body temperature: {bt_max}°C"
                })
            if bt_min < THRESHOLDS["body_temp_min"] and bt_min > 10:
                anomalies.append({
                    "type":    "low_body_temp",
                    "value":   bt_min,
                    "message": f"Low body temperature: {bt_min}°C"
                })

    logger.info(f"Anomalies detected: {len(anomalies)}")
    for a in anomalies:
        logger.info(f"  [{a['type']}] {a['message']}")

    return anomalies


# ─────────────────────────────────────────
# STEP 3: RAG SETUP
# ─────────────────────────────────────────
def build_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)

    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        logger.info("Loading existing vectorstore...")
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    logger.info("Building new vectorstore...")
    docs = []
    for pattern in ["**/*.txt", "**/*.pdf", "**/*.json", "**/*.md"]:
        for f in glob.glob(os.path.join(RAG_DIR, pattern), recursive=True):
            try:
                loader = PyPDFLoader(f) if f.endswith(".pdf") else TextLoader(f, encoding="utf-8")
                docs.extend(loader.load())
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=64,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Split into {len(chunks)} chunks")

    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    vs.persist()
    return vs


def build_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.7}
    )


# ─────────────────────────────────────────
# STEP 4: GENERATE QUESTIONS VIA GEMMA4
# ─────────────────────────────────────────
QUESTION_PROMPT = PromptTemplate(
    input_variables=["anomalies", "context"],
    template="""You are a caring health assistant for elderly patients.
Based on the sensor anomalies detected and the health context below,
generate 3 short, clear, caring questions to ask the senior patient.
Questions should be simple, easy to understand, and directly related to the anomalies.
Return ONLY the questions, one per line, no numbering, no extra text.

Sensor Anomalies Detected:
{anomalies}

Health Context from knowledge base:
{context}

Questions:"""
)


def generate_questions(anomalies, retriever):
    if not anomalies:
        return ["How are you feeling today?",
                "Did you sleep well last night?",
                "Is there anything uncomfortable you would like to tell me?"]

    # Build anomaly summary
    anomaly_text = "\n".join([f"- {a['message']}" for a in anomalies])

    # RAG retrieval based on anomaly types
    query = " ".join([a["type"].replace("_", " ") for a in anomalies])
    docs  = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Generate questions via Gemma4
    llm = OllamaLLM(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
        temperature=0.3
    )

    prompt_text = QUESTION_PROMPT.format(
        anomalies=anomaly_text,
        context=context
    )

    logger.info("Generating questions via Gemma4...")
    response = llm.invoke(prompt_text)

    # Parse questions
    questions = [q.strip() for q in response.strip().split("\n") if q.strip()]
    questions = questions[:5]   # max 5 questions

    logger.info(f"Generated {len(questions)} questions")
    return questions


# ─────────────────────────────────────────
# STEP 5: PIPER TTS → SPEAKER
# ─────────────────────────────────────────
def speak(text):
    logger.info(f"Speaking: {text[:60]}...")
    print(f"\n🔊 Assistant: {text}")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name

    try:
        proc = subprocess.Popen(
            [PIPER_BIN, "--model", PIPER_MODEL, "--output_file", tmp_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        proc.communicate(input=text.encode("utf-8"))

        subprocess.run(
            ["aplay", "-D", SPK_DEVICE, tmp_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ─────────────────────────────────────────
# STEP 6: MIC RECORDING
# ─────────────────────────────────────────
def record_from_mic():
    p = pyaudio.PyAudio()
    stream = p.open(
        format=MIC_FORMAT,
        channels=MIC_CHANNELS,
        rate=MIC_RATE,
        input=True,
        frames_per_buffer=MIC_CHUNK
    )

    print("🎤 Listening...")
    frames = []
    silent_chunks    = 0
    max_silent_chunks = int(SILENCE_DURATION * MIC_RATE / MIC_CHUNK)
    max_chunks        = int(MAX_RECORD_SEC   * MIC_RATE / MIC_CHUNK)
    speaking_started  = False

    for _ in range(max_chunks):
        data = stream.read(MIC_CHUNK, exception_on_overflow=False)
        frames.append(data)

        amplitude = np.abs(np.frombuffer(data, dtype=np.int16)).mean()

        if amplitude > SILENCE_THRESHOLD:
            speaking_started = True
            silent_chunks = 0
        elif speaking_started:
            silent_chunks += 1

        if speaking_started and silent_chunks > max_silent_chunks:
            break

    stream.stop_stream()
    stream.close()
    p.terminate()

    if not frames:
        return None

    # Stereo → Mono → Resample to 16kHz
    audio = np.frombuffer(b"".join(frames), dtype=np.int16)
    audio = audio.reshape(-1, 2)[:, 0]
    audio = signal.resample_poly(audio, TARGET_RATE, MIC_RATE)
    return audio.astype(np.int16)


# ─────────────────────────────────────────
# STEP 7: WHISPER STT
# ─────────────────────────────────────────
def transcribe(audio_array, whisper_model):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name

    with wave.open(tmp_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_RATE)
        wf.writeframes(audio_array.tobytes())

    result = whisper_model.transcribe(tmp_path, language="en")
    os.unlink(tmp_path)
    return result["text"].strip()


# ─────────────────────────────────────────
# STEP 8: LOG DIALOG
# ─────────────────────────────────────────
def log_dialog(role, text):
    """Log dialog entry to dialog.log"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"{timestamp} | {role.upper():10s} | {text}"

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry + "\n")


def log_session_start(anomalies):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n" + "="*70 + "\n")
        f.write(f"SESSION START: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"ACCOUNT: {ACCOUNT} | DATE: {DATE}\n")
        f.write(f"ANOMALIES: {len(anomalies)}\n")
        for a in anomalies:
            f.write(f"  [{a['type']}] {a['message']}\n")
        f.write("="*70 + "\n\n")


def log_session_end():
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\nSESSION END: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")


# ─────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────
def main():
    print("="*60)
    print("  Senior Health Dialog System")
    print("="*60)

    # Force CPU for Whisper
    torch.cuda.is_available = lambda: False

    # Load Whisper
    print("\n⏳ Loading Whisper...")
    whisper_model = whisper.load_model(WHISPER_MODEL, device="cpu")
    print("✅ Whisper ready")

    # Build RAG
    print("⏳ Loading RAG vectorstore...")
    vectorstore = build_vectorstore()
    retriever   = build_retriever(vectorstore)
    print("✅ RAG ready")

    # Fetch sensor data
    print("\n⏳ Fetching sensor data...")
    sensor_data = fetch_all_sensors()

    # Detect anomalies
    anomalies = detect_anomalies(sensor_data)

    # Log session start
    log_session_start(anomalies)

    # Generate questions
    print("\n⏳ Generating dialog questions...")
    questions = generate_questions(anomalies, retriever)

    print(f"\n✅ {len(questions)} questions ready")
    print(f"✅ Logging to: {LOG_FILE}")
    print("\n" + "─"*60)

    # Greeting
    greeting = "Hello! I am your health assistant. I have reviewed your health data from last night and have a few questions for you."
    speak(greeting)
    log_dialog("ASSISTANT", greeting)

    # Ask each question and record answer
    import time
    current_type = None

    MAX_QUESTIONS = 5
    questions = questions[:MAX_QUESTIONS]
    for i, item in enumerate(questions, 1):
        # handle both tuple (type, question) and plain string
        if isinstance(item, tuple):
            anomaly_type, question = item[0], item[1]
        else:
            anomaly_type, question = "general", item

        print(f"\n[Question {i}/{len(questions)}]")

        # Print anomaly section header when type changes
        if anomaly_type != current_type:
            current_type = anomaly_type
            section = anomaly_type.replace("_", " ").upper()
            print(f"\n── {section} ──")
            log_dialog("SYSTEM", f"Section: {section}")

        # Speak question
        speak(question)
        log_dialog("ASSISTANT", question)

        # Record answer
        input("⏎  Press Enter when ready to answer...")
        audio = record_from_mic()

        if audio is None:
            logger.warning("No audio captured")
            log_dialog("SENIOR", "[No response - no audio]")
            speak("I did not catch that. Let us move on.")
            continue

        # Transcribe
        answer = transcribe(audio, whisper_model)
        if not answer:
            log_dialog("SENIOR", "[No response - silence]")
            speak("I did not hear anything. Let us continue.")
            continue

        print(f"👤 Senior: {answer}")
        log_dialog("SENIOR", answer)
        time.sleep(0.5)

    # Closing
    closing = "Thank you for answering my questions. I will share this information with your care team. Please take care and have a good day!"
    speak(closing)
    log_dialog("ASSISTANT", closing)

    # Log session end
    log_session_end()

    print(f"\n✅ Dialog complete! Log saved to: {LOG_FILE}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted")
        log_dialog("SYSTEM", "Session interrupted by user")
        log_session_end()
        sys.exit(0)
