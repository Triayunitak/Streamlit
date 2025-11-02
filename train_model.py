import streamlit as st
import pandas as pd
import numpy as np
import json
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
from prophet.serialize import model_to_json
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Prediksi NO₂ Mingguan", layout="wide")
st.title("📈 Prediksi NO₂ Mingguan")
st.markdown(
    "Aplikasi ini memprediksi konsentrasi NO₂ berdasarkan data historis sebelum dan setelah COVID-19."
)

# --- 1. Load Data ---
@st.cache_data
def load_openeo_json_to_df(filename):
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
    df = df.sort_values('date').reset_index(drop=True)
    return df

st.info("Memuat data...")
before_df = load_openeo_json_to_df('no2_before_covid.json')
after_df = load_openeo_json_to_df('no2_after_covid.json')
df = pd.concat([before_df, after_df])
df['value'] = df['value'].interpolate()
df = df.dropna()
df = df.set_index('date')

# --- 2. Resample Mingguan ---
st.info("Mengubah data menjadi rata-rata mingguan...")
df_weekly = df['value'].resample('W-MON').mean().interpolate()
df_prophet = df_weekly.reset_index().rename(columns={'date': 'ds', 'value': 'y'})

st.subheader("Data Mingguan")
st.dataframe(df_prophet.tail(10))  # tampilkan 10 baris terakhir

# --- 3. Training / Test Split ---
test_size = st.slider("Jumlah minggu untuk test set", min_value=4, max_value=20, value=10)
train_data = df_prophet.iloc[:-test_size]
test_data = df_prophet.iloc[-test_size:]

st.write(f"Ukuran training: {len(train_data)}, Ukuran test: {len(test_data)}")

# --- 4. Latih Model ---
st.info("Melatih model Prophet...")
model = Prophet(
    weekly_seasonality=False,
    yearly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='multiplicative'
)
model.fit(train_data)
# --- 5. Evaluasi ---
future_test = model.make_future_dataframe(periods=len(test_data), freq='W-MON')
forecast_test = model.predict(future_test)
predictions = forecast_test['yhat'].iloc[-len(test_data):]
mape = mean_absolute_percentage_error(test_data['y'], predictions)
st.metric("MAPE Test Set", f"{mape*100:.2f}%")

if mape*100 < 10:
    st.success("🎯 Target MAPE < 10% terpenuhi!")
else:
    st.warning(f"⚠️ Target MAPE < 10% belum terpenuhi ({mape*100:.2f}%).")

# --- 6. Prediksi ke Depan ---
weeks_ahead = st.number_input("Jumlah minggu ke depan untuk prediksi", min_value=1, max_value=52, value=12)

if st.button("Prediksi ke Depan"):
    final_model = Prophet(
        weekly_seasonality=False,
        yearly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    final_model.fit(df_prophet)
    
    future = final_model.make_future_dataframe(periods=weeks_ahead, freq='W-MON')
    forecast = final_model.predict(future)
    
    st.subheader("📊 Grafik Prediksi NO₂")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df_prophet['ds'], df_prophet['y'], label="Data Asli")
    ax.plot(forecast['ds'], forecast['yhat'], label="Prediksi", linestyle='--')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='orange', alpha=0.2)
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("NO₂ (µg/m³)")
    ax.legend()
    st.pyplot(fig)
    
    st.subheader("📋 Tabel Prediksi")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(weeks_ahead))
    
    # --- Simpan Model ---
    with open('prophet_model_weekly.json', 'w') as fout:
        fout.write(model_to_json(final_model))
    st.success("✅ Model telah disimpan untuk deployment.")