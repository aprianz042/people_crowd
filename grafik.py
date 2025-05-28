import streamlit as st
import time
import random
import pandas as pd

st.title("Grafik Realtime Jumlah Orang")

placeholder = st.empty()

# Simulasi data realtime (ganti dengan data count hasil deteksi)
data = pd.DataFrame({'Jumlah Orang': []})

for _ in range(100):
    new_count = random.randint(0, 20)  # ganti dengan hasil deteksi real kamu
    new_row = pd.DataFrame({'Jumlah Orang': [new_count]})
    data = pd.concat([data, new_row], ignore_index=True)

    # tampilkan grafik update realtime
    placeholder.line_chart(data)

    time.sleep(0.5)
