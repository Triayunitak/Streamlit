import pandas as pd
import numpy as np
import json
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
from prophet.serialize import model_to_json
import warnings
import matplotlib.pyplot as plt
import os

# --- Sembunyikan Warnings ---
warnings.filterwarnings("ignore")
plt.style.use('seaborn-darkgrid')  # tampilan grafik lebih bagus

# --- 1. Fungsi Load Data ---
def load_openeo_json_to_df(filename):
    print(f"📄 Membaca file JSON: {filename}")
    with open(filename, 'r') as f:
        data = json.load(f)
    dates, values = [], []
    for date_str, value_list in data.items():
        dates.append(date_str)
        if value_list and value_list[0] and value_list[0][0] is not None:
            values.append(value_list[0][0])
        else:
            values.append(np.nan)
    df = pd.DataFrame({'date': dates, 'value': values})
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    df = df.sort_values(by='date').reset_index(drop=True)
    return df

# --- 2. Load Data & Preview ---
print("🧹 Memuat data sebelum dan sesudah COVID...")
before_df = load_openeo_json_to_df('no2_before_covid.json')
after_df = load_openeo_json_to_df('no2_after_covid.json')

df = pd.concat([before_df, after_df])
df["value"] = df["value"].interpolate(method="linear")
df = df.dropna()
df = df.set_index('date')

print(f"✅ Total data harian: {len(df)} baris")
print("\n📊 Statistik Data:")
display(df.describe())

# --- 3. Resample Mingguan ---
print("\n⏱ Mengubah data menjadi rata-rata mingguan untuk mengurangi noise...")
df_weekly = df['value'].resample('W-MON').mean().interpolate(method="linear")

plt.figure(figsize=(12,5))
plt.plot(df_weekly.index, df_weekly.values, color='dodgerblue', linewidth=2)
plt.title("📈 Rata-rata Mingguan NO₂", fontsize=16)
plt.xlabel("Tanggal")
plt.ylabel("NO₂ (µg/m³)")
plt.grid(True, alpha=0.3)
plt.show()

df_prophet = df_weekly.reset_index().rename(columns={'date': 'ds', 'value': 'y'})
print(f"✅ Data mingguan siap: {len(df_prophet)} baris")

# --- 4. Train/Test Split ---
test_size = 10
train_data = df_prophet.iloc[:-test_size]
test_data = df_prophet.iloc[-test_size:]

print(f"\n📌 Ukuran data train: {len(train_data)}, test: {len(test_data)}")

# --- 5. Latih Model Prophet ---
print("\n⚙️ Melatih model Prophet dengan musiman tahunan...")
model = Prophet(
    weekly_seasonality=False,
    yearly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='multiplicative'
)
model.fit(train_data)
print("✅ Model selesai dilatih")

# --- 6. Evaluasi Model ---
future_test = model.make_future_dataframe(periods=len(test_data), freq='W-MON')
forecast_test = model.predict(future_test)
predictions = forecast_test['yhat'].iloc[-len(test_data):]
mape = mean_absolute_percentage_error(test_data['y'], predictions)
print(f"📏 MAPE pada test set: {mape*100:.2f}%")

# --- Visualisasi Test vs Prediksi ---
plt.figure(figsize=(12,5))
plt.plot(train_data['ds'], train_data['y'], label='Train', color='blue')
plt.plot(test_data['ds'], test_data['y'], label='Test', color='green')
plt.plot(test_data['ds'], predictions, label='Prediksi', color='red', linestyle='--')
plt.title("📊 Prediksi vs Real (Test Set)", fontsize=16)
plt.xlabel("Tanggal")
plt.ylabel("NO₂ (µg/m³)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# --- 7. Latih Ulang Model Seluruh Data ---
print("\n🔄 Melatih ulang model pada seluruh data untuk deployment...")
final_model = Prophet(
    weekly_seasonality=False,
    yearly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='multiplicative'
)
final_model.fit(df_prophet)

# --- 8. Simpan Model ---
os.makedirs("models", exist_ok=True)
with open('models/prophet_model_weekly.json', 'w') as fout:
    fout.write(model_to_json(final_model))

print("✅ Model tersimpan di 'models/prophet_model_weekly.json'")
print("\n🎉 Selesai! Data siap untuk deployment dan prediksi ke depan.")
