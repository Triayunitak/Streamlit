import streamlit as st
import pandas as pd
import numpy as np
import json
from prophet import Prophet
from prophet.serialize import model_to_json
from sklearn.metrics import mean_absolute_percentage_error
from datetime import datetime, timedelta
import os

# --- Sembunyikan warnings ---
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Prediksi NO₂ Mingguan", layout="wide")
st.title("🌿 Prediksi Konsentrasi NO₂ Mingguan")

st.markdown("""
Aplikasi ini memprediksi konsentrasi NO₂ untuk beberapa minggu ke depan menggunakan model Prophet.
Pastikan file JSON NO₂ tersedia di folder proyek.
""")

# --- Upload file JSON ---
before_file = st.file_uploader("Unggah NO₂ sebelum COVID (JSON)", type="json")
after_file = st.file_uploader("Unggah NO₂ setelah COVID (JSON)", type="json")

if before_file and after_file:
    # --- Fungsi load JSON ---
    def load_openeo_json_to_df(uploaded_file):
        data = json.load(uploaded_file)
        dates, values = [], []
        for date_str, value_list in data.items():
            dates.append(date_str)
            if value_list and value_list[0] and value_list[0][0] is not None:
                values.append(value_list[0][0])
            else:
                values.append(np.nan)
        df = pd.DataFrame({'ds': dates, 'y': values})
        df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df = df.sort_values('ds').dropna().reset_index(drop=True)
        return df

    df_before = load_openeo_json_to_df(before_file)
    df_after = load_openeo_json_to_df(after_file)
    df = pd.concat([df_before, df_after]).sort_values('ds').reset_index(drop=True)

    # --- Resample mingguan ---
    df.set_index('ds', inplace=True)
    df_weekly = df['y'].resample('W-MON').mean().interpolate()
    df_weekly = df_weekly.reset_index().rename(columns={'y':'y','ds':'ds'})

    st.subheader("📊 Data NO₂ Mingguan")
    st.dataframe(df_weekly.tail(10))

    # --- Input prediksi ---
    n_weeks = st.number_input("Jumlah minggu prediksi", min_value=1, max_value=52, value=4)

    # --- Latih dan prediksi ---
    if st.button("Prediksi"):
        model = Prophet(
            weekly_seasonality=False,
            yearly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        model.fit(df_weekly)

        future = model.make_future_dataframe(periods=n_weeks, freq='W-MON')
        forecast = model.predict(future)

        # --- Tampilkan hasil ---
        pred_df = forecast[['ds', 'yhat']].tail(n_weeks)
        pred_df = pred_df.rename(columns={'ds':'Tanggal', 'yhat':'Prediksi NO₂ (µg/m³)'})

        st.subheader("📈 Hasil Prediksi")
        st.line_chart(pred_df.set_index('Tanggal'))
        st.dataframe(pred_df.style.format({"Prediksi NO₂ (µg/m³)": "{:.2f}"}))

        # --- Simpan model ---
        os.makedirs("models", exist_ok=True)
        with open('models/prophet_model_weekly.json', 'w') as fout:
            fout.write(model_to_json(model))
        st.success("✅ Model berhasil dilatih dan disimpan di 'models/prophet_model_weekly.json'")