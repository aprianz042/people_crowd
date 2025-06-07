import streamlit as st
import cv2
import yt_dlp
from ultralytics import YOLO
import numpy as np
import time
import pandas as pd
from datetime import datetime, time as dt_time
import joblib

# --- Model prediksi jumlah orang (berbasis waktu)
model_pred = joblib.load("model.pkl")
date_input = datetime.today().date()
waktu = [dt_time(h, m) for h in range(24) for m in (0, 30)]

def prediksi(date_input):
    rows = []
    for t in waktu:
        dt = datetime.combine(date_input, t)
        hour = dt.hour
        minute = dt.minute
        dayofweek = dt.weekday()
        is_weekend = dayofweek >= 5
        is_peak_hour = ((hour in range(7, 9) or hour in range(17, 20)) if not is_weekend else (hour in range(16, 21)))
        rows.append({
            "timestamp": dt,
            "hour": hour,
            "minute": minute,
            "dayofweek": dayofweek,
            "is_weekend": is_weekend,
            "is_peak_hour": is_peak_hour
        })
    df_pred = pd.DataFrame(rows)
    features = ['hour', 'minute', 'dayofweek', 'is_weekend', 'is_peak_hour']
    df_pred['predicted_count'] = model_pred.predict(df_pred[features])
    df_pred = df_pred.set_index('timestamp')
    return df_pred['predicted_count']

# --- Init session state langsung True
st.session_state.run_detection = True

# --- Fungsi ambil stream URL dari YouTube Live
def get_youtube_stream_url(youtube_url):
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'format': 'best',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        formats = info.get('formats', [])
        for f in formats:
            if f.get('protocol') == 'm3u8':
                return f['url']
        return info.get("url", None)

# --- Load YOLOv8
model_detect = YOLO("yolov8n.pt")

# --- URL YouTube Hardcoded
YOUTUBE_URL = "https://www.youtube.com/watch?v=gFRtAAmiFbE"

# --- UI
st.title("People Detection dari YouTube Live (Auto Start Tanpa Tombol)")

kol1, kol2, kol3 = st.columns(3)
with kol1:
    DT_WINDOW = st.empty()
with kol2:
    placeholder_chart = st.empty()
with kol3:
    placeholder_count = st.empty()
    time_placeholder = st.empty()

pred_placeholder = st.empty()
data = pd.DataFrame(columns=['Waktu', 'Jumlah Orang'])

# --- Proses Deteksi Otomatis
if st.session_state.run_detection:
    pred_placeholder.line_chart(prediksi(date_input))
    stream_url = get_youtube_stream_url(YOUTUBE_URL)
    if stream_url:
        cap = cv2.VideoCapture(stream_url)

        if not cap.isOpened():
            st.error("Gagal membuka stream video.")
        else:
            st.success("Deteksi dimulai otomatis...")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.warning("Frame tidak terbaca.")
                    break

                results = model_detect(frame)
                boxes = results[0].boxes
                count = 0

                height, width = frame.shape[:2]
                black_frame = np.zeros((height, width, 3), dtype=np.uint8)

                for box in boxes:
                    cls = int(box.cls[0])
                    if model_detect.model.names[cls] == "person":
                        count += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        cv2.circle(black_frame, (cx, cy), radius=15, color=(255, 255, 255), thickness=-1)

                DT_WINDOW.image(cv2.cvtColor(black_frame, cv2.COLOR_BGR2RGB), width=600)

                now = datetime.now().strftime("%H:%M:%S")
                new_row = pd.DataFrame({'Waktu': [now], 'Jumlah Orang': [count]})
                data = pd.concat([data, new_row], ignore_index=True)
                display_data = data.copy()
                display_data.set_index('Waktu', inplace=True)
                display_data = display_data.tail(40)

                placeholder_count.metric(label="Jumlah Orang Saat Ini", value=count)
                time_placeholder.metric(label="Waktu", value=now)
                placeholder_chart.line_chart(display_data)

                time.sleep(0.05)

            cap.release()
    else:
        st.error("Gagal mendapatkan URL streaming dari YouTube.")
