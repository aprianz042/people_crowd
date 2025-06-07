import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import time
import joblib
import psutil

# --- Load model prediksi jumlah orang (berbasis waktu)
model_pred = joblib.load("model.pkl")

# --- Load YOLOv8
model_detect = YOLO("yolov8n.pt")

# --- Init session state
if "run_detection" not in st.session_state:
    st.session_state.run_detection = True

# --- Layout
kol1, kol2 = st.columns(2, border=True)
with kol1:
    kol1.markdown('People Detection')
    FRAME_WINDOW = st.empty()
with kol2:
    kol2.markdown('Performance')
    cpu_placeholder = st.empty()
    ram_placeholder = st.empty()
    total_time_placeholder = st.empty()  # placeholder untuk total waktu
    avg_cpu_placeholder = st.empty()
    avg_ram_placeholder = st.empty()

# --- Proses Deteksi
if st.session_state.run_detection:
    cap = cv2.VideoCapture("people.mp4")

    if not cap.isOpened():
        st.error("Gagal membuka file video 'people.mp4'. Pastikan file ada di direktori.")
    else:
        true_fps = cap.get(cv2.CAP_PROP_FPS)
        start_time = time.time()

        cpu_sum = 0.0
        ram_sum = 0.0
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Video selesai atau frame tidak terbaca.")
                break

            results = model_detect(frame)
            boxes = results[0].boxes

            height, width = frame.shape[:2]
            black_frame = np.zeros((height, width, 3), dtype=np.uint8)

            for box in boxes:
                cls = int(box.cls[0])
                if model_detect.model.names[cls] == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    cv2.circle(black_frame, (cx, cy), radius=15, color=(255, 255, 255), thickness=-1)

            FRAME_WINDOW.image(cv2.cvtColor(black_frame, cv2.COLOR_BGR2RGB))

            cpu_percent = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()
            cpu_placeholder.metric(label="CPU Usage (%)", value=f"{cpu_percent:.0f}%")
            ram_placeholder.metric(label="RAM Usage (%)", value=f"{memory_info.percent:.0f}%")

             # Akumulasi untuk rata-rata
            cpu_sum += cpu_percent
            ram_sum += memory_info.percent
            frame_count += 1

        cap.release()

        end_time = time.time()
        total_duration = end_time - start_time

        total_time_placeholder.markdown(f"**Total waktu pemrosesan video:** {total_duration:.2f} detik (~{total_duration/60:.2f} menit)")

        if frame_count > 0:
            avg_cpu = cpu_sum / frame_count
            avg_ram = ram_sum / frame_count
            avg_cpu_placeholder.markdown(f"**Rata-rata CPU Usage:** {avg_cpu:.2f}%")
            avg_ram_placeholder.markdown(f"**Rata-rata RAM Usage:** {avg_ram:.2f}%")