#!/usr/bin/env python3
"""
RAG Voice Pipeline
- MIC (card 2, plughw:2,0) → Whisper STT
- RAG (nomic-embed-text + Chroma + reranker) → Gemma4:e2b
- Piper TTS → Speaker (card 0, plughw:0,0)
"""

import os
import sys
import time
import wave
import struct
import tempfile
import subprocess
import numpy as np
import scipy.signal as signal
import pyaudio
import whisper
import torch
torch.cuda.is_available = lambda: False

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import glob

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

# Audio - MIC
MIC_DEVICE      = "plughw:2,0"
MIC_CHANNELS    = 2
MIC_RATE        = 44100
MIC_FORMAT      = pyaudio.paInt16
MIC_CHUNK       = 1024
TARGET_RATE     = 16000         # Whisper needs 16000Hz

# Audio - Speaker
SPK_DEVICE      = "plughw:0,0"
SPK_CHANNELS    = 1
SPK_RATE        = 16000

# Silence detection
SILENCE_THRESHOLD   = 500       # amplitude threshold
SILENCE_DURATION    = 2.0       # seconds of silence to stop recording
MAX_RECORD_SECONDS  = 30        # max recording time

# RAG
RAG_DIR     = "/home/soo/Work/mom-i/senior/rag"
CHROMA_DIR  = "/home/soo/Work/mom-i/senior/chroma_db"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL   = "gemma4:e2b"
OLLAMA_URL  = "http://localhost:11434"

# Piper TTS
PIPER_MODEL = "/home/soo/Work/mom-i/senior/piper/en_US-amy-medium.onnx"
PIPER_BIN   = "piper"

# Whisper model size: tiny / base / small / medium
WHISPER_MODEL = "base"


# ─────────────────────────────────────────
# STEP 1: MIC RECORDING WITH SILENCE DETECT
# ─────────────────────────────────────────

def record_from_mic():
    """Record from MIC until silence detected, return 16kHz mono numpy array."""
    p = pyaudio.PyAudio()

    stream = p.open(
        format=MIC_FORMAT,
        channels=MIC_CHANNELS,
        rate=MIC_RATE,
        input=True,
        frames_per_buffer=MIC_CHUNK
    )

    print("🎤 Listening... (speak now, auto-stop on silence)")

    frames = []
    silent_chunks = 0
    max_silent_chunks = int(SILENCE_DURATION * MIC_RATE / MIC_CHUNK)
    max_chunks = int(MAX_RECORD_SECONDS * MIC_RATE / MIC_CHUNK)
    speaking_started = False

    for _ in range(max_chunks):
        data = stream.read(MIC_CHUNK, exception_on_overflow=False)
        frames.append(data)

        # check amplitude
        audio_data = np.frombuffer(data, dtype=np.int16)
        amplitude = np.abs(audio_data).mean()

        if amplitude > SILENCE_THRESHOLD:
            speaking_started = True
            silent_chunks = 0
        else:
            if speaking_started:
                silent_chunks += 1

        if speaking_started and silent_chunks > max_silent_chunks:
            print("🔇 Silence detected, processing...")
            break

    stream.stop_stream()
    stream.close()
    p.terminate()

    if not frames:
        return None

    # Convert to numpy
    raw = b"".join(frames)
    audio = np.frombuffer(raw, dtype=np.int16)

    # Stereo → Mono (left channel)
    audio = audio.reshape(-1, 2)[:, 0]

    # Resample 44100 → 16000
    audio = signal.resample_poly(audio, TARGET_RATE, MIC_RATE)
    audio = audio.astype(np.int16)

    return audio


# ─────────────────────────────────────────
# STEP 2: WHISPER STT
# ─────────────────────────────────────────

def transcribe(audio_array, whisper_model):
    """Transcribe audio numpy array using Whisper."""
    print("📝 Transcribing...")

    # Save to temp wav file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name

    with wave.open(tmp_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # 16-bit = 2 bytes
        wf.setframerate(TARGET_RATE)
        wf.writeframes(audio_array.tobytes())

    result = whisper_model.transcribe(tmp_path, language="en")
    os.unlink(tmp_path)

    text = result["text"].strip()
    print(f"🗣️  You said: {text}")
    return text


# ─────────────────────────────────────────
# STEP 3: RAG - LOAD DOCUMENTS
# ─────────────────────────────────────────

def load_documents():
    docs = []

    txt_files = glob.glob(os.path.join(RAG_DIR, "**/*.txt"), recursive=True)
    for f in txt_files:
        loader = TextLoader(f, encoding="utf-8")
        docs.extend(loader.load())

    pdf_files = glob.glob(os.path.join(RAG_DIR, "**/*.pdf"), recursive=True)
    for f in pdf_files:
        loader = PyPDFLoader(f)
        docs.extend(loader.load())

    json_files = glob.glob(os.path.join(RAG_DIR, "**/*.json"), recursive=True)
    for f in json_files:
        loader = TextLoader(f, encoding="utf-8")
        docs.extend(loader.load())

    print(f"✅ Loaded {len(docs)} documents")
    return docs


# ─────────────────────────────────────────
# STEP 4: RAG - BUILD VECTORSTORE
# ─────────────────────────────────────────

def build_vectorstore():
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_URL
    )

    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print("📂 Loading existing vectorstore...")
        return Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )

    print("📂 Building new vectorstore...")
    docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ Split into {len(chunks)} chunks")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    vectorstore.persist()
    print(f"✅ Vectorstore saved")
    return vectorstore


# ─────────────────────────────────────────
# STEP 5: RAG - RETRIEVER + RERANKER
# ─────────────────────────────────────────

def build_retriever(vectorstore):
    # MMR retriever - diversity + relevance, no extra packages needed
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,           # return top 3
            "fetch_k": 10,    # fetch 10 candidates first
            "lambda_mult": 0.7  # 0=max diversity, 1=max relevance
        }
    )
    print("✅ MMR Retriever ready")
    return retriever


# ─────────────────────────────────────────
# STEP 6: GEMMA4 STREAMING GENERATION
# ─────────────────────────────────────────

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a caring assistant for elderly patients.
Use the following context to answer the question clearly and simply.
Keep your answer concise and easy to understand.

Context:
{context}

Question:
{question}

Answer:"""
)

def generate_response(retriever, query):
    """Retrieve + rerank + generate with streaming, return full response text."""

    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt_text = PROMPT.format(context=context, question=query)

    print(f"\n📄 Retrieved {len(docs)} chunks after reranking")
    print(f"{'─'*50}")
    print(f"🤖 Gemma4 Response:\n")

    llm = OllamaLLM(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
        temperature=0.3,
    )

    full_response = ""
    for chunk in llm.stream(prompt_text):
        print(chunk, end="", flush=True)
        full_response += chunk

    print(f"\n{'─'*50}\n")
    return full_response


# ─────────────────────────────────────────
# STEP 7: PIPER TTS → SPEAKER
# ─────────────────────────────────────────

def speak(text):
    """Convert text to speech using Piper and play via speaker."""
    print("🔊 Speaking...")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name

    # Generate speech with Piper
    cmd = [
        PIPER_BIN,
        "--model", PIPER_MODEL,
        "--output_file", tmp_path
    ]

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    process.communicate(input=text.encode("utf-8"))

    # Play via speaker (plughw:0,0)
    play_cmd = [
        "aplay",
        "-D", SPK_DEVICE,
        tmp_path
    ]
    subprocess.run(play_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    os.unlink(tmp_path)
    print("✅ Done speaking")


# ─────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────

def main():
    print("=" * 50)
    print("  RAG Voice Pipeline")
    print("  MIC → Whisper → RAG → Gemma4 → Piper → Speaker")
    print("=" * 50)

    # Load Whisper
    print(f"\n⏳ Loading Whisper ({WHISPER_MODEL})...")
    whisper_model = whisper.load_model(WHISPER_MODEL, device="cpu")
    print("✅ Whisper ready")

    # Build RAG
    vectorstore = build_vectorstore()
    retriever = build_retriever(vectorstore)

    print("\n✅ All systems ready!")
    print("Say something to start (or Ctrl+C to quit)\n")

    # Speak intro
    speak("Hello! I am your health assistant. Please speak your question.")

    while True:
        try:
            print("─" * 50)
            input("⏎  Press Enter to start speaking (or Ctrl+C to quit)...")

            # Record from MIC
            audio = record_from_mic()
            if audio is None:
                print("⚠️  No audio captured, try again")
                continue

            # Whisper STT
            query = transcribe(audio, whisper_model)
            if not query:
                print("⚠️  Could not transcribe, try again")
                speak("Sorry, I could not understand. Please try again.")
                continue

            # RAG + Gemma4
            response = generate_response(retriever, query)

            # Piper TTS → Speaker
            speak(response)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            speak("Goodbye! Take care.")
            sys.exit(0)

        except Exception as e:
            print(f"❌ Error: {e}")
            continue


if __name__ == "__main__":
    main()
