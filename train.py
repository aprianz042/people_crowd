import pandas as pd
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import joblib

# ====== 1. Baca file CSV ======
df = pd.read_csv("dummy_crowd_weather_2years.csv", parse_dates=["timestamp"])

# ====== 2. Feature Engineering ======
df['hour'] = df['timestamp'].dt.hour
df['minute'] = df['timestamp'].dt.minute
df['dayofweek'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['dayofweek'] >= 5
df['is_peak_hour'] = (
    ((df['hour'].between(7, 8)) | (df['hour'].between(17, 19))) & (~df['is_weekend'])
) | (
    df['hour'].between(16, 20) & df['is_weekend']
)

# One-hot encoding untuk 'condition'
condition_dummies = pd.get_dummies(df['condition'], prefix='cond')
df = pd.concat([df, condition_dummies], axis=1)

# ====== 3. Training Model ======
features = ['hour', 'minute', 'dayofweek', 'is_weekend', 'is_peak_hour'] + list(condition_dummies.columns)
X = df[features]
y = df['count']

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

model = XGBRegressor()
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")
print("Model berhasil disimpan ke 'model.pkl'")

# Evaluasi
y_pred = model.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.2f}")

# ====== 4. Prediksi Jumlah Orang ======
# Contoh prediksi untuk Senin, 26 Mei 2025, pukul 18:30 dengan kondisi "Clear"
future_time = datetime(2025, 5, 26, 19, 30)
future_condition = "Clear"

future_data = {
    'hour': [future_time.hour],
    'minute': [future_time.minute],
    'dayofweek': [future_time.weekday()],
    'is_weekend': [future_time.weekday() >= 5],
    'is_peak_hour': [
        (future_time.hour in range(7, 9) or future_time.hour in range(17, 20)) if future_time.weekday() < 5
        else (future_time.hour in range(16, 21))
    ]
}

# Tambahkan semua kolom kondisi dengan 0, kecuali yang sesuai kondisi cuaca
for col in condition_dummies.columns:
    future_data[col] = [1 if col == f'cond_{future_condition}' else 0]

future_df = pd.DataFrame(future_data)
prediction = model.predict(future_df)
print(f"Prediksi jumlah orang pada {future_time.strftime('%A, %d %B %Y %H:%M')} dengan kondisi {future_condition} adalah sekitar {prediction[0]:.0f} orang.")
