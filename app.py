import cv2
import joblib
import psutil  
import os
import pymysql
import requests
import random
import yt_dlp
import numpy as np
import streamlit as st
import pandas as pd
import google.generativeai as genai
from ultralytics import YOLO
from dotenv import load_dotenv
from datetime import datetime, time as dt_time

host=os.getenv("DB_HOST")       
user=os.getenv("DB_USER")       
password=os.getenv("DB_GEMBOK")   
database=os.getenv("DB_NYA")
gem_api=os.getenv("GOOGLE_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# --- Load YOLOv8
model_detect = YOLO("yolov8n.pt")

client = genai.configure(api_key=gem_api)

# --- URL YouTube
#YOUTUBE_URL = "https://www.youtube.com/watch?v=gFRtAAmiFbE"
YOUTUBE_URL = "https://www.youtube.com/live/W-Y_Xp_wQgw?si=Wqy0bekdjmQoqOlD"

# --- Model prediksi jumlah orang (berbasis waktu)
model_pred = joblib.load("model.pkl")
date_input = datetime.today().date()
waktu = [dt_time(h, m) for h in range(24) for m in (0, 30)]

def insert_data(t, c, w, temp, humi): 
    conn = pymysql.connect(host=host, user=user, password=password, database=database)   
    cursor = conn.cursor()
    sql = "INSERT INTO traffic_data (timestamp, count, `condition`, temp, humidity) VALUES (%s, %s, %s, %s, %s)"
    val = (t, c, w, temp, humi)
    cursor.execute(sql, val)
    conn.commit()
    print(cursor.rowcount, "record inserted.")
    conn.close()

def get_weather(api_key, lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather = {
            "temp": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "condition": data["weather"][0]["main"],
            "desc": data["weather"][0]["description"],
            "icon": data["weather"][0]["icon"]
        }
        return weather
    else:
        return None
    
def map_weather_desc_to_int(desc):
    weather_description_to_int = {
        "clear sky": 0,
        "few clouds": 1,
        "scattered clouds": 2,
        "broken clouds": 3,
        "overcast clouds": 4,

        "light rain": 5,
        "moderate rain": 6,
        "heavy intensity rain": 7,
        "very heavy rain": 8,
        "extreme rain": 9,
        "freezing rain": 10,
        "light intensity shower rain": 11,
        "shower rain": 12,
        "heavy intensity shower rain": 13,
        "ragged shower rain": 14,

        "thunderstorm with light rain": 15,
        "thunderstorm with rain": 16,
        "thunderstorm with heavy rain": 17,
        "light thunderstorm": 18,
        "thunderstorm": 19,
        "heavy thunderstorm": 20,
        "ragged thunderstorm": 21,
        "thunderstorm with light drizzle": 22,
        "thunderstorm with drizzle": 23,
        "thunderstorm with heavy drizzle": 24,

        "light snow": 25,
        "snow": 26,
        "heavy snow": 27,
        "sleet": 28,
        "light shower sleet": 29,
        "shower sleet": 30,
        "light rain and snow": 31,
        "rain and snow": 32,
        "light shower snow": 33,
        "shower snow": 34,
        "heavy shower snow": 35,

        "mist": 36,
        "smoke": 37,
        "haze": 38,
        "sand/dust whirls": 39,
        "fog": 40,
        "sand": 41,
        "dust": 42,
        "volcanic ash": 43,
        "squalls": 44,
        "tornado": 45
    }

    return weather_description_to_int.get(desc.lower(), -1)  # -1 jika deskripsi tidak dikenal

def condition_to_icon(condition):
    mapping = {
        "Clear": "☀️",
        "Clouds": "☁️",
        "Rain": "🌧️",
        "Drizzle": "🌦️",
        "Thunderstorm": "⛈️",
        "Snow": "❄️",
        "Mist": "🌫️",
        "Smoke": "🌫️",
        "Haze": "🌫️",
        "Dust": "🌫️",
        "Fog": "🌫️",
        "Sand": "🌫️",
        "Ash": "🌫️",
        "Squall": "🌬️",
        "Tornado": "🌪️"
    }
    return mapping.get(condition, "🌈")  # default: 🌈 jika kondisi tidak dikenal

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
    df_pred['predicted_count'] = df_pred['predicted_count'].round().astype(int)
    df_pred = df_pred.set_index('timestamp')
    return df_pred['predicted_count']

def prediksi_dengan_kondisi_cuaca(date_input) -> pd.DataFrame:
    model = model_pred
    kondisi_cuaca = ['Clear', 'Clouds', 'Rain', 'Drizzle', 'Thunderstorm', 'Mist']
    suhu_per_kondisi = {
        'Clear': (28, 34),
        'Clouds': (25, 30),
        'Rain': (22, 27),
        'Drizzle': (23, 28),
        'Thunderstorm': (21, 26),
        'Mist': (20, 25),
    }

    rows = []
    for w in waktu:
        dt = datetime.combine(date_input, w)
        hour = dt.hour
        minute = dt.minute
        dayofweek = dt.weekday()
        is_weekend = dayofweek >= 5
        is_peak_hour = ((hour in range(7, 9) or hour in range(17, 20)) and not is_weekend) or \
                       (hour in range(16, 21) and is_weekend)

        condition = random.choices(
            population=kondisi_cuaca,
            weights=[0.4, 0.3, 0.1, 0.1, 0.05, 0.05],
            k=1
        )[0]

        temp_range = suhu_per_kondisi[condition]
        temperature = round(random.uniform(*temp_range), 1)  # satu angka di belakang koma

        rows.append({
            "timestamp": dt,
            "hour": hour,
            "minute": minute,
            "dayofweek": dayofweek,
            "is_weekend": is_weekend,
            "is_peak_hour": is_peak_hour,
            "condition": condition,
            "temperature": temperature
        })

    df = pd.DataFrame(rows)

    # One-hot encode kondisi cuaca, sesuai nama fitur saat training model
    # Gunakan prefix 'cond_' sesuai nama fitur di model
    df_encoded = pd.get_dummies(df['condition'], prefix='cond')

    # Gabungkan one-hot encoded ke df utama
    df = pd.concat([df, df_encoded], axis=1)

    # Fitur lengkap sesuai yang model harapkan
    features = [
        'hour', 'minute', 'dayofweek', 'is_weekend', 'is_peak_hour',
        'cond_Clear', 'cond_Clouds', 'cond_Drizzle', 'cond_Fog',
        'cond_Haze', 'cond_Mist', 'cond_Rain', 'cond_Thunderstorm'
    ]

    # Pastikan semua kolom fitur ada di df, kalau tidak ada buat kolom dengan 0
    for col in features:
        if col not in df.columns:
            df[col] = 0

    # Urutkan kolom fitur sesuai urutan yang model inginkan
    df_features = df[features]

    # Prediksi dengan model
    df['count_pred'] = model.predict(df_features).round().astype(int)

    df_pred = df.set_index('timestamp')

    return df_pred[['count_pred']], df[['timestamp', 'count_pred', 'condition', 'temperature']]

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

# --- Init session state
st.session_state.run_detection = True

if "genai_model" not in st.session_state:
        st.session_state["genai_model"] = genai.GenerativeModel('gemini-2.0-flash')

def get_gemini_response(question):
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(question)
    return response.text

# --- UI Layout
st.title("Multimedia IoT - People Crowd Detection")

kol1, kol2 = st.columns(2, border=True)
with kol1:
    kol1.markdown('People Detection')
    FRAME_WINDOW = st.empty()
with kol2:
    kol2.markdown('Visualization')
    DT_WINDOW = st.empty()

col_1, col_2, col_3 = st.columns([1, 1, 2], border=True)
with col_1:
    col_1.markdown('RealTime Count')
    placeholder_count = st.empty()
    time_placeholder = st.empty()
with col_2:
    col_2.markdown('Weather Monitor')
    cuaca = st.empty()
    suhu = st.empty()
    kelembaban = st.empty()
with col_3:
    col_3.markdown('Realtime Chart')
    placeholder_chart = st.empty()

kolom1, kolom2 = st.columns(2, border=True)
with kolom1:
    st.markdown(f'Prediksi keramaian hari ini {date_input}')
    pred_placeholder = st.empty()    
with kolom2:
    st.markdown(f'Rekomendasi kunjungan hari ini {date_input}')
    #cuaca_placeholder = st.empty()
    rekom_placeholder = st.empty()

data = pd.DataFrame(columns=['Waktu', 'Jumlah Orang'])
a, b = prediksi_dengan_kondisi_cuaca(date_input)

d_frame = b.to_string()

aturan = [
    f"""
    Tugas:
    Buatkan dua rekomendasi jam kunjungan terbaik untuk orang yang introvert dan extrovert dari skema prediksi kepadatan orang dan cuaca pada dataframe berikut:

    \nSkema Dataframe:
    {d_frame}
    
    \njawabannya jangan ada kalimat pembuka,
    \nhasilnya cukup 2 poin atau 2 paragraf,
    \nperhatikan juga faktor temp dan codition nya dalam memberi rekomendasi, 
    \nwilayah waktunya adalah GMT +8, namun jangan disebutkan di jawabannya,
    \nhasil rekomendasi hanya dalam rentang dari jam saat ini sampai jam 00:00,
    \nberikan jam yang tidak disarankan juga
    """
]

# --- Proses Deteksi
if st.session_state.run_detection:

    query = get_gemini_response(aturan)

    pred_placeholder.bar_chart(a)
    rekom_placeholder.markdown(query)
    #cuaca_placeholder.dataframe(b)
    #pred_placeholder.bar_chart(prediksi(date_input))

    
    stream_url = get_youtube_stream_url(YOUTUBE_URL)

    MALIOBORO_LAT, MALIOBORO_LON = -7.791594544974983, 110.3662733746616

    if stream_url:
        cap = cv2.VideoCapture(stream_url)

        if not cap.isOpened():
            st.error("Gagal membuka stream video.")
        else:

            weather = get_weather(OPENWEATHER_API_KEY, MALIOBORO_LAT, MALIOBORO_LON)
            emoji = condition_to_icon(weather["condition"])
            desc_cuaca = weather['condition']
            desc_cuaca = f'{emoji} {desc_cuaca}'

            d_suhu = weather['temp']
            d_suhu = f'{d_suhu} °C'

            d_kelembaban = weather['humidity']
            d_kelembaban = f'{d_kelembaban} %'

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
                        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width=600)
                DT_WINDOW.image(cv2.cvtColor(black_frame, cv2.COLOR_BGR2RGB), width=600)

                skr = datetime.now()
                now_str = skr.strftime("%H:%M:%S")
                count_list.append(count)

                now = datetime.now().strftime("%H:%M:%S")
                new_row = pd.DataFrame({'Waktu': [now], 'Jumlah Orang': [count]})
                data = pd.concat([data, new_row], ignore_index=True)
                display_data = data.copy()
                display_data.set_index('Waktu', inplace=True)
                display_data = display_data.tail(40)

                # --- Sistem Metrik
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_info = psutil.virtual_memory()

                placeholder_count.metric(label="Jumlah Orang Saat Ini", value=count)
                time_placeholder.metric(label="Waktu", value=now)
                cuaca.metric(label="Kondisi", value=desc_cuaca)
                suhu.metric(label="Suhu", value=d_suhu)
                kelembaban.metric(label="Kelembaban", value=d_kelembaban)                
                placeholder_chart.line_chart(display_data)

                #time.sleep(0.05)

                #fungsi SAVE to DB
                if skr.minute == 30 or skr.minute == 00 or skr.minute == 37:
                    current_hour_minute = (skr.hour, skr.minute)
                    if last_insert_hour_minute != current_hour_minute:
                        if count_list:
                            avg_count = int(np.mean(count_list))
                            timestamp_str = skr
                            insert_data(timestamp_str, avg_count, weather['condition'], weather['temp'], weather['humidity'])
                            count_list.clear()
                            last_insert_hour_minute = current_hour_minute
                else:
                    last_insert_hour_minute = None

            cap.release()
    else:
        st.error("Gagal mendapatkan URL streaming dari YouTube.")
