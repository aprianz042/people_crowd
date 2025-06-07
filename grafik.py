import streamlit as st
import cv2
import yt_dlp
from ultralytics import YOLO
import numpy as np
import time
import pandas as pd
from datetime import datetime, time as dt_time
import joblib
import psutil  
import os
from dotenv import load_dotenv
import pymysql

st.title("Grafik Realtime Jumlah Orang")

placeholder = st.empty()

# Simulasi data realtime (ganti dengan data count hasil deteksi)
data = pd.DataFrame({'Jumlah Orang': []})

model_detect = YOLO("yolov8n.pt")

# --- URL YouTube
YOUTUBE_URL = "https://www.youtube.com/watch?v=gFRtAAmiFbE"

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

pred_placeholder = st.empty()

data = pd.DataFrame(columns=['Waktu', 'Jumlah Orang'])
placeholder_chart = st.empty()

# --- Proses Deteksi
if st.session_state.run_detection:
    stream_url = get_youtube_stream_url(YOUTUBE_URL)
    if stream_url:
        cap = cv2.VideoCapture(stream_url)

        if not cap.isOpened():
            st.error("Gagal membuka stream video.")
        else:
            last_insert_hour_minute = None
            count_list = []
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
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                skr = datetime.now()
                now_str = skr.strftime("%H:%M:%S")
                count_list.append(count)

                now = datetime.now().strftime("%H:%M:%S")
                new_row = pd.DataFrame({'Waktu': [now], 'Jumlah Orang': [count]})
                data = pd.concat([data, new_row], ignore_index=True)
                display_data = data.copy()
                display_data.set_index('Waktu', inplace=True)
                display_data = display_data.tail(40)

                placeholder_chart.line_chart(display_data)

                #time.sleep(0.05)
            cap.release()
    else:
        st.error("Gagal mendapatkan URL streaming dari YouTube.")
