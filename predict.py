import streamlit as st
import pandas as pd
from datetime import datetime, time
import joblib

# Load model
model = joblib.load("model.pkl")

# Input tanggal
date_input = st.date_input("Pilih tanggal", datetime.today().date())
#date_input = datetime.today().date()

label = (f'Prediksi Jumlah Orang per 30 Menit pada {date_input}')
st.title(label)

# Generate waktu tiap 30 menit sepanjang hari
times = [time(h, m) for h in range(24) for m in (0, 30)]

# Buat dataframe input untuk prediksi
rows = []
for t in times:
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

# Prediksi
features = ['hour', 'minute', 'dayofweek', 'is_weekend', 'is_peak_hour']
df_pred['predicted_count'] = model.predict(df_pred[features])

# Set timestamp jadi index biar chart lebih rapi
df_pred = df_pred.set_index('timestamp')

# Tampilkan line chart interaktif
st.line_chart(df_pred['predicted_count'])
