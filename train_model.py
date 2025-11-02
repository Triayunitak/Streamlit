import streamlit as st
import pandas as pd
import numpy as np
import json
from prophet import Prophet
from prophet.serialize import model_to_json
from datetime import datetime, timedelta
import os

# --- Sembunyikan warnings ---
import warnings
warnings.filterwarnings("ignore")

# --- Fungsi Load Data ---
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
    df = pd.DataFrame({'ds': pd.to_datetime(dates), 'y': values})
    df = df.sort_values(by='ds').reset_index(drop=True)
    df['y'] = df['y'].interpolate(method='linear')
    return df

# --- Load Data ---
st.title("Prediksi NO₂ Mingguan Interaktif")
st.markdown("Masukkan parameter prediksi di samping, lalu klik **Prediksi**.")

before_df = load_openeo_json_to_df("no2_before_covid.json")
after_df = load_openeo_json_to_df("no2_after_covid.json")
df = pd.concat([before_df, after_df]).reset_index(drop=True)

# --- Sidebar Input ---
st.sidebar.header("Pengaturan Prediksi")
start_date = st.sidebar.date_input("Tanggal mulai prediksi", datetime.today())
n_weeks = st.sidebar.number_input("Jumlah minggu prediksi", min_value=1, max_value=52, value=4)
seasonality_mode = st.sidebar.selectbox("Mode Musiman", ["additive", "multiplicative"])

# --- Prediksi Button ---
if st.sidebar.button("Prediksi"):
    # Latih model Prophet pada seluruh data
    model = Prophet(
        weekly_seasonality=False,
        yearly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode=seasonality_mode
    )
    model.fit(df)

    # Buat future dataframe
    future = model.make_future_dataframe(periods=n_weeks, freq='W-MON')
    forecast = model.predict(future)

    # Filter hasil prediksi yang baru
    pred_df = forecast[forecast['ds'] >= pd.to_datetime(start_date)][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].reset_index(drop=True)

    # --- Tampilkan Grafik ---
    st.subheader("📈 Grafik Prediksi NO₂")
    st.line_chart(pred_df.set_index('ds')['yhat'])

    # --- Tampilkan Tabel ---
    st.subheader("📋 Tabel Hasil Prediksi")
    st.dataframe(pred_df.style.format({
        'yhat': '{:.2f}',
        'yhat_lower': '{:.2f}',
        'yhat_upper': '{:.2f}'
    }))

    # --- Simpan Model untuk Deployment ---
    os.makedirs("models", exist_ok=True)
    with open('models/prophet_model_weekly.json', 'w') as fout:
        fout.write(model_to_json(model))
    st.success("✅ Prediksi selesai dan model disimpan.")