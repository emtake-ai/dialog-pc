#!/usr/bin/env python3
"""
Sensor Data Fetcher from emtake API
Fetches: SleepData, Temp, Breath, dB, IndoorTemp, ALL
"""

import requests
import json
from datetime import datetime

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
API_URL  = "http://relay.emtake.com/api/query"
ACCOUNT  = "test10@test.com"       # change to your account
DATE     = datetime.now().strftime("%Y-%m-%d")  # today's date

HEADERS  = {
    "Content-Type": "application/json"
}

# ─────────────────────────────────────────
# FETCH SINGLE CMD
# ─────────────────────────────────────────
def fetch_sensor(cmd, account=ACCOUNT, date=DATE, val="1"):
    payload = {
        "Type":    "LLMREPORT",
        "Account": account,
        "CMD":     cmd,
        "val":     val,
        "date":    date
    }

    try:
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        return data

    except requests.exceptions.ConnectionError:
        print(f"❌ Connection error for {cmd}")
        return None
    except requests.exceptions.Timeout:
        print(f"❌ Timeout for {cmd}")
        return None
    except Exception as e:
        print(f"❌ Error fetching {cmd}: {e}")
        return None


# ─────────────────────────────────────────
# FETCH ALL SENSOR DATA
# ─────────────────────────────────────────
def fetch_all_sensors(account=ACCOUNT, date=DATE):
    print(f"\n{'='*55}")
    print(f"  Fetching Sensor Data")
    print(f"  Account : {account}")
    print(f"  Date    : {date}")
    print(f"{'='*55}\n")

    sensor_data = {}

    # 1. Sleep Data
    print("⏳ Fetching SleepData...")
    data = fetch_sensor("SleepData", account, date)
    if data:
        sensor_data["sleep"] = data
        print(f"  ✅ SleepData: {json.dumps(data, indent=2, ensure_ascii=False)[:200]}...")
    else:
        print("  ⚠️  No SleepData")

    # 2. Temperature
    print("\n⏳ Fetching Temp...")
    data = fetch_sensor("Temp", account, date)
    if data:
        sensor_data["temp"] = data
        print(f"  ✅ Temp: {json.dumps(data, ensure_ascii=False)}")
    else:
        print("  ⚠️  No Temp data")

    # 3. Breathing
    print("\n⏳ Fetching Breath...")
    data = fetch_sensor("Breath", account, date)
    if data:
        sensor_data["breath"] = data
        print(f"  ✅ Breath: {json.dumps(data, ensure_ascii=False)}")
    else:
        print("  ⚠️  No Breath data")

    # 4. Noise (dB)
    print("\n⏳ Fetching dB...")
    data = fetch_sensor("dB", account, date)
    if data:
        sensor_data["db"] = data
        print(f"  ✅ dB: {json.dumps(data, ensure_ascii=False)}")
    else:
        print("  ⚠️  No dB data")

    # 5. Indoor Temperature
    print("\n⏳ Fetching IndoorTemp...")
    data = fetch_sensor("IndoorTemp", account, date)
    if data:
        sensor_data["indoor_temp"] = data
        print(f"  ✅ IndoorTemp: {json.dumps(data, ensure_ascii=False)}")
    else:
        print("  ⚠️  No IndoorTemp data")

    return sensor_data


# ─────────────────────────────────────────
# FETCH ALL IN ONE CALL
# ─────────────────────────────────────────
def fetch_all_in_one(account=ACCOUNT, date=DATE):
    print("\n⏳ Fetching ALL sensors in one call...")
    data = fetch_sensor("ALL", account, date)
    if data:
        print(f"  ✅ ALL: {json.dumps(data, indent=2, ensure_ascii=False)[:300]}...")
    return data


# ─────────────────────────────────────────
# PARSE ANOMALIES
# ─────────────────────────────────────────
def parse_response(raw):
    """Parse API response - may be string or dict."""
    import json
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except:
            return {}
    return raw if raw else {}


def detect_anomalies(sensor_data):
    anomalies = []
    DATE = datetime.now().strftime("%Y-%m-%d")
    PREV = (datetime.now() - __import__("datetime").timedelta(days=1)).strftime("%Y-%m-%d")

    # ── Sleep anomalies ──
    sleep_raw = parse_response(sensor_data.get("sleep", {}))
    # find TEST key (TEST10 etc)
    test_key = next((k for k in sleep_raw if k.startswith("TEST")), None)
    if test_key:
        sleep_today = sleep_raw[test_key].get(DATE) or sleep_raw[test_key].get(PREV, {})
        sessions    = sleep_today.get("sessions", [])
        total_wakeup = sum(s.get("wake_up", 0) for s in sessions)
        total_dur    = sum(s.get("duration_min", 0) for s in sessions)
        print(f"\n🛏️  Sleep: sessions={len(sessions)}, total_wakeup={total_wakeup}, duration={total_dur}min")

        if total_wakeup > 3:
            anomalies.append({
                "type": "frequent_waking",
                "value": total_wakeup,
                "message": f"Woke up {total_wakeup} times during sleep"
            })
        if total_dur < 360:  # less than 6 hours
            anomalies.append({
                "type": "short_sleep",
                "value": total_dur,
                "message": f"Short sleep duration: {total_dur} minutes"
            })

    # ── Breath anomalies ──
    breath_raw = parse_response(sensor_data.get("breath", {}))
    test_key = next((k for k in breath_raw if k.startswith("TEST")), None)
    if test_key:
        breath_today = breath_raw[test_key].get(DATE) or breath_raw[test_key].get(PREV, {})
        b_min = breath_today.get("Min", 0)
        b_max = breath_today.get("Max", 0)
        print(f"🫁  Breath: min={b_min}, max={b_max}")
        if b_max > 20 or b_min < 8:
            anomalies.append({
                "type": "abnormal_breathing",
                "value": f"{b_min}-{b_max}",
                "message": f"Abnormal breathing rate detected: {b_min}-{b_max} breaths/min"
            })

    # ── Indoor Temp anomalies ──
    indoor_raw = parse_response(sensor_data.get("indoor_temp", {}))
    test_key = next((k for k in indoor_raw if k.startswith("TEST")), None)
    if test_key:
        indoor_today = indoor_raw[test_key].get(DATE) or indoor_raw[test_key].get(PREV, {})
        t_min = indoor_today.get("Min", 0)
        t_max = indoor_today.get("Max", 0)
        print(f"🌡️  IndoorTemp: min={t_min}, max={t_max}")
        if t_max > 28 or t_min < 16:
            anomalies.append({
                "type": "abnormal_room_temp",
                "value": f"{t_min}-{t_max}",
                "message": f"Room temperature out of range: {t_min}-{t_max}°C"
            })

    # ── dB anomalies ──
    db_raw = parse_response(sensor_data.get("db", {}))
    test_key = next((k for k in db_raw if k.startswith("TEST")), None)
    if test_key:
        db_today = db_raw[test_key].get(DATE) or db_raw[test_key].get(PREV, {})
        db_max = db_today.get("Max", 0)
        print(f"🔊  dB: max={db_max}")
        if db_max > 60:
            anomalies.append({
                "type": "high_noise",
                "value": db_max,
                "message": f"High noise level during sleep: {db_max}dB"
            })

    # ── Body Temp anomalies ──
    temp_raw = parse_response(sensor_data.get("temp", {}))
    test_key = next((k for k in temp_raw if k.startswith("TEST")), None)
    if test_key:
        temp_today = temp_raw[test_key].get(DATE) or temp_raw[test_key].get(PREV, {})
        t_min = temp_today.get("Min", 0)
        t_max = temp_today.get("Max", 0)
        print(f"🌡️  BodyTemp: min={t_min}, max={t_max}")
        if t_max > 37.5:
            anomalies.append({
                "type": "high_body_temp",
                "value": t_max,
                "message": f"High body temperature: {t_max}°C"
            })
        if t_min < 35.0:
            anomalies.append({
                "type": "low_body_temp",
                "value": t_min,
                "message": f"Low body temperature: {t_min}°C"
            })

    return anomalies


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    # Fetch all sensor data
    sensor_data = fetch_all_sensors()

    # Also try ALL in one call
    all_data = fetch_all_in_one()
    if all_data:
        sensor_data["all"] = all_data

    # Detect anomalies
    print(f"\n{'─'*55}")
    print("🔍 Detecting anomalies...")
    anomalies = detect_anomalies(sensor_data)

    if anomalies:
        print(f"\n⚠️  Found {len(anomalies)} anomalies:")
        for a in anomalies:
            print(f"   - [{a['type']}] {a['message']}")
    else:
        print("✅ No anomalies detected")

    # Save to file for RAG pipeline
    output = {
        "date":       DATE,
        "account":    ACCOUNT,
        "sensor_data": sensor_data,
        "anomalies":  anomalies,
        "fetched_at": datetime.now().isoformat()
    }

    with open("/tmp/sensor_data.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved to /tmp/sensor_data.json")

    return output


if __name__ == "__main__":
    main()
