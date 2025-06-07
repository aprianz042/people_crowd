import streamlit as st
import cv2
import yt_dlp
import pymysql
import requests
import numpy as np
import pandas as pd
import joblib
import psutil
import os
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
from ultralytics import YOLO
from datetime import datetime, time as dt_time
import logging
import time
from contextlib import contextmanager
import threading
import queue

# --- KONFIGURASI
YOUTUBE_URL = "https://www.youtube.com/live/W-Y_Xp_wQgw?si=Wqy0bekdjmQoqOlD"
OPENWEATHER_API_KEY = "b3a7fc753e5585d726234e2088421c5b" 
MALIOBORO_LAT, MALIOBORO_LON = -7.791594544974983, 110.3662733746616

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- ENV & DB SETUP
load_dotenv()

# Validasi environment variables
required_env_vars = ["DB_HOST", "DB_USER", "DB_GEMBOK", "DB_NYA"]
for var in required_env_vars:
    if not os.getenv(var):
        st.error(f"Environment variable {var} tidak ditemukan!")
        st.stop()

host = os.getenv("DB_HOST")
user = os.getenv("DB_USER")
password = os.getenv("DB_GEMBOK")
database = os.getenv("DB_NYA")

# Database connection pool manager
@contextmanager
def get_db_connection():
    """Context manager untuk koneksi database"""
    conn = None
    try:
        conn = pymysql.connect(
            host=host, 
            user=user, 
            password=password, 
            database=database,
            autocommit=True,
            connect_timeout=10,
            read_timeout=10
        )
        yield conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def insert_data(timestamp, avg_person, avg_motorcycle, avg_car, avg_bus):
    """Insert data ke database dengan error handling"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            sql = """INSERT INTO traffic_data (timestamp, person, motorcycle, car, bus) 
                     VALUES (%s, %s, %s, %s, %s)"""
            val = (timestamp, avg_person, avg_motorcycle, avg_car, avg_bus)
            cursor.execute(sql, val)
            logger.info(f"Data inserted successfully: {timestamp}")
    except Exception as e:
        logger.error(f"Failed to insert data: {e}")
        st.error(f"Gagal menyimpan data: {e}")

# --- DATA CUACA
@st.cache_data(ttl=600)  # Cache selama 10 menit
def get_weather():
    """Ambil data cuaca dengan error handling"""
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={MALIOBORO_LAT}&lon={MALIOBORO_LON}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise exception untuk status code 4xx/5xx
        data = response.json()
        
        cuaca = data['weather'][0]['main']
        deskripsi = data['weather'][0]['description']
        suhu = data['main']['temp']
        icon = data['weather'][0]['icon']
        icon_url = f"http://openweathermap.org/img/wn/{icon}@2x.png"
        
        return cuaca, deskripsi, suhu, icon_url
    except requests.RequestException as e:
        logger.error(f"Weather API error: {e}")
        return "N/A", "Data cuaca tidak tersedia", "N/A", ""
    except KeyError as e:
        logger.error(f"Weather data parsing error: {e}")
        return "N/A", "Format data cuaca tidak valid", "N/A", ""

# --- Model prediksi
@st.cache_resource
def load_prediction_model():
    """Load model prediksi dengan caching"""
    try:
        return joblib.load("model.pkl")
    except FileNotFoundError:
        st.error("File model.pkl tidak ditemukan!")
        return None
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

def prediksi(date_input, model):
    """Generate prediksi dengan validasi model"""
    if model is None:
        return pd.Series(dtype=int)
    
    waktu = [dt_time(h, m) for h in range(24) for m in (0, 30)]
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
    
    try:
        df_pred['predicted_count'] = model.predict(df_pred[features])
        df_pred['predicted_count'] = df_pred['predicted_count'].round().astype(int)
        df_pred = df_pred.set_index('timestamp')
        return df_pred['predicted_count']
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return pd.Series(dtype=int)

# --- Fungsi ambil stream URL dari YouTube Live
@st.cache_data(ttl=3600)  # Cache selama 1 jam
def get_youtube_stream_url(youtube_url):
    """Ambil URL stream YouTube dengan caching"""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'format': 'best[height<=720]',  # Batasi resolusi untuk performa
        'socket_timeout': 30,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            formats = info.get('formats', [])
            
            # Prioritas format m3u8
            for f in formats:
                if f.get('protocol') == 'm3u8':
                    return f['url']
            
            return info.get("url", None)
    except Exception as e:
        logger.error(f"YouTube stream extraction error: {e}")
        return None

# --- Load YOLOv8
@st.cache_resource
def load_yolo_model():
    """Load YOLO model dengan caching"""
    try:
        return YOLO("yolov8n.pt")
    except Exception as e:
        st.error(f"Gagal memuat model YOLO: {e}")
        return None

# --- Inisialisasi session state
def init_session_state():
    """Inisialisasi session state"""
    if "run_detection" not in st.session_state:
        st.session_state.run_detection = True
    if "detection_data" not in st.session_state:
        st.session_state.detection_data = pd.DataFrame(columns=['Waktu', 'Person', 'Motorcycle', 'Car', 'Bus'])
    if "count_lists" not in st.session_state:
        st.session_state.count_lists = {
            'person': [],
            'motorcycle': [],
            'car': [],
            'bus': []
        }
    if "last_insert_time" not in st.session_state:
        st.session_state.last_insert_time = None
    if "chart_counter" not in st.session_state:
        st.session_state.chart_counter = 0

def process_detection_results(results, frame):
    """Process YOLO detection results"""
    boxes = results[0].boxes
    count_dict = {"person": 0, "motorcycle": 0, "car": 0, "bus": 0}
    
    height, width = frame.shape[:2]
    visualization_frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    for box in boxes:
        cls = int(box.cls[0])
        class_name = results[0].names[cls]
        
        if class_name in count_dict:
            count_dict[class_name] += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            # Color mapping
            colors = {
                "person": (0, 255, 0),
                "motorcycle": (255, 0, 0),
                "car": (0, 0, 255),
                "bus": (255, 255, 0)
            }
            color = colors.get(class_name, (255, 255, 255))
            
            # Draw on visualization frame
            cv2.circle(visualization_frame, (cx, cy), radius=15, color=color, thickness=-1)
            
            # Draw bounding box on original frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return count_dict, visualization_frame

def update_data_storage(count_dict):
    """Update data storage and handle database insertion"""
    now = datetime.now()
    time_str = now.strftime("%H:%M:%S")
    
    # Add to session data
    new_row = pd.DataFrame({
        'Waktu': [time_str],
        'Person': [count_dict["person"]],
        'Motorcycle': [count_dict["motorcycle"]],
        'Car': [count_dict["car"]],
        'Bus': [count_dict["bus"]],
    })
    
    st.session_state.detection_data = pd.concat([st.session_state.detection_data, new_row], ignore_index=True)
    
    # Keep only last 100 records for memory efficiency
    if len(st.session_state.detection_data) > 100:
        st.session_state.detection_data = st.session_state.detection_data.tail(100).reset_index(drop=True)
    
    # Update count lists for averaging
    for key in st.session_state.count_lists:
        st.session_state.count_lists[key].append(count_dict[key])
    
    # Database insertion logic (every 30 minutes)
    if now.minute == 30:
        current_time_key = (now.hour, now.minute)
        if st.session_state.last_insert_time != current_time_key:
            save_to_database()
            st.session_state.last_insert_time = current_time_key
    else:
        st.session_state.last_insert_time = None

def save_to_database():
    """Save averaged data to database"""
    try:
        if any(st.session_state.count_lists[key] for key in st.session_state.count_lists):
            averages = {}
            for key in st.session_state.count_lists:
                if st.session_state.count_lists[key]:
                    averages[key] = int(np.mean(st.session_state.count_lists[key]))
                else:
                    averages[key] = 0
            
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            insert_data(timestamp_str, averages['person'], averages['motorcycle'], 
                       averages['car'], averages['bus'])
            
            # Clear lists after saving
            for key in st.session_state.count_lists:
                st.session_state.count_lists[key].clear()
                
            st.success("Data berhasil disimpan ke database!")
    except Exception as e:
        logger.error(f"Error saving to database: {e}")

# --- UI Layout
def main():
    """Main application function"""
    st.set_page_config(
        page_title="Digital Twin Malioboro",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    init_session_state()
    
    st.title("🏛️ Digital Twin Malioboro - People & Traffic Monitoring")
    st.caption("Crowd & vehicle monitoring berbasis live camera Malioboro, Yogyakarta.")
    
    # Load models
    model_pred = load_prediction_model()
    model_detect = load_yolo_model()
    
    if model_detect is None:
        st.error("Model YOLO tidak dapat dimuat. Aplikasi dihentikan.")
        return
    
    # Weather info
    with st.expander("🌤️ Info Cuaca Malioboro (Realtime)", expanded=True):
        cuaca, deskripsi, suhu, icon_url = get_weather()
        col1, col2 = st.columns([1, 3])
        if icon_url and "N/A" not in icon_url:
            col1.image(icon_url, width=80)
        col2.markdown(f"**{cuaca.title()}** ({deskripsi})")
        col2.markdown(f"**Suhu:** {suhu}°C")
    
    # Main layout
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('### 📹 Live Detection')
        frame_window = st.empty()
    with col2:
        st.markdown('### 🎯 Visualization')
        viz_window = st.empty()
    
    # Metrics and charts
    col_1, col_2, col_3 = st.columns([1, 1, 2])
    with col_1:
        st.markdown('### 📊 Real-time Count')
        metric_placeholders = {
            'person': st.empty(),
            'motorcycle': st.empty(),
            'car': st.empty(),
            'bus': st.empty(),
            'time': st.empty()
        }
    
    with col_2:
        st.markdown('### 💻 System Resources')
        resource_placeholders = {
            'cpu': st.empty(),
            'ram': st.empty()
        }
    
    with col_3:
        st.markdown('### 📈 Live Charts')
        chart_placeholder = st.empty()
    
    # Prediction chart
    st.markdown('### 🔮 Prediksi Keramaian Hari Ini')
    if model_pred:
        date_input = datetime.today().date()
        prediction = prediksi(date_input, model_pred)
        if not prediction.empty:
            # Create a more detailed prediction chart
            pred_fig = go.Figure()
            pred_fig.add_trace(go.Bar(
                x=prediction.index.strftime('%H:%M'),
                y=prediction.values,
                name='Predicted Count',
                marker_color='rgba(55, 126, 184, 0.7)'
            ))
            pred_fig.update_layout(
                title="Hourly Crowd Prediction",
                xaxis_title="Time",
                yaxis_title="Predicted People Count",
                height=400,
                margin=dict(t=50, l=0, r=0, b=30)
            )
            st.plotly_chart(pred_fig, use_container_width=True, key="prediction_chart")
        else:
            st.warning("Prediksi tidak dapat dibuat.")
    else:
        st.warning("Model prediksi tidak tersedia.")
    
    # Control buttons
    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("▶️ Start Detection", type="primary"):
            st.session_state.run_detection = True
    with col_stop:
        if st.button("⏹️ Stop Detection"):
            st.session_state.run_detection = False
    
    # Detection loop
    if st.session_state.run_detection:
        run_detection(model_detect, frame_window, viz_window, metric_placeholders, 
                     resource_placeholders, chart_placeholder)

def run_detection(model_detect, frame_window, viz_window, metric_placeholders, 
                 resource_placeholders, chart_placeholder):
    """Run the detection loop"""
    stream_url = get_youtube_stream_url(YOUTUBE_URL)
    
    if not stream_url:
        st.error("Gagal mendapatkan URL streaming dari YouTube.")
        return
    
    cap = cv2.VideoCapture(stream_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for real-time processing
    
    if not cap.isOpened():
        st.error("Gagal membuka stream video.")
        return
    
    frame_count = 0
    process_every_n_frames = 3  # Process every 3rd frame for better performance
    
    try:
        while cap.isOpened() and st.session_state.run_detection:
            ret, frame = cap.read()
            if not ret:
                st.warning("Frame tidak terbaca. Mencoba reconnect...")
                time.sleep(1)
                continue
            
            frame_count += 1
            
            # Process only every nth frame
            if frame_count % process_every_n_frames == 0:
                # Run detection
                results = model_detect(frame, verbose=False)
                count_dict, viz_frame = process_detection_results(results, frame)
                
                # Update displays
                frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width=600)
                viz_window.image(cv2.cvtColor(viz_frame, cv2.COLOR_BGR2RGB), width=600)
                
                # Update metrics
                now = datetime.now().strftime("%H:%M:%S")
                metric_placeholders['person'].metric("🧑 Person", count_dict["person"])
                metric_placeholders['motorcycle'].metric("🏍️ Motorcycle", count_dict["motorcycle"])
                metric_placeholders['car'].metric("🚗 Car", count_dict["car"])
                metric_placeholders['bus'].metric("🚌 Bus", count_dict["bus"])
                metric_placeholders['time'].metric("⏰ Time", now)
                
                # System resources
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_info = psutil.virtual_memory()
                resource_placeholders['cpu'].metric("💾 CPU Usage", f"{cpu_percent:.1f}%")
                resource_placeholders['ram'].metric("🖥️ RAM Usage", f"{memory_info.percent:.1f}%")
                
                # Update data storage
                update_data_storage(count_dict)
                
                # Update charts
                update_charts(chart_placeholder, count_dict)
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.1)
            
    except Exception as e:
        logger.error(f"Detection loop error: {e}")
        st.error(f"Error dalam deteksi: {e}")
    finally:
        cap.release()

def update_charts(chart_placeholder, count_dict):
    """Update the charts with current data"""
    if len(st.session_state.detection_data) > 0:
        # Increment chart counter for unique keys
        st.session_state.chart_counter += 1
        counter = st.session_state.chart_counter
        
        display_data = st.session_state.detection_data.tail(20).copy()
        
        with chart_placeholder.container():
            # Pie chart for current distribution
            st.markdown("#### 🥧 Current Detection Distribution")
            pie_labels = ['Person', 'Motorcycle', 'Car', 'Bus']
            pie_values = [count_dict["person"], count_dict["motorcycle"], count_dict["car"], count_dict["bus"]]
            
            pie_fig = go.Figure()
            pie_fig.add_trace(go.Pie(
                labels=pie_labels, 
                values=pie_values, 
                hole=0.4,
                textinfo='label+percent',
                textposition='inside'
            ))
            pie_fig.update_layout(
                showlegend=True,
                height=300,
                margin=dict(t=20, l=0, r=0, b=0)
            )
            st.plotly_chart(pie_fig, use_container_width=True, key=f"pie_chart_{counter}")
            
            # Line chart for trends
            if len(display_data) > 1:
                st.markdown("#### 📈 Detection Trends (Last 20 Records)")
                display_data_indexed = display_data.set_index('Waktu')
                
                line_fig = px.line(
                    display_data_indexed,
                    y=['Person', 'Motorcycle', 'Car', 'Bus'],
                    title="Detection Trends Over Time",
                    labels={"value": "Count", "index": "Time"},
                    markers=True
                )
                line_fig.update_layout(
                    height=300,
                    margin=dict(t=30, l=0, r=0, b=0),
                    legend_title_text='Categories'
                )
                st.plotly_chart(line_fig, use_container_width=True, key=f"line_chart_{counter}")

if __name__ == "__main__":
    main()