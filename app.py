import streamlit as st
import pandas as pd
import numpy as np
import json
from prophet import Prophet
from prophet.serialize import model_from_json
import plotly.graph_objects as go

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi Kualitas Udara NO2",
    page_icon="💨",
    layout="wide"
)

# --- Fungsi Helper ---
@st.cache_data
def load_historical_data_weekly():
    def load_openeo_json_to_df(filename):
        with open(filename, 'r') as f: data = json.load(f)
        dates, values = [], []
        for date_str, value_list in data.items():
            dates.append(date_str)
            if value_list and value_list[0] and value_list[0][0] is not None:
                values.append(value_list[0][0])
            else: values.append(np.nan)
        df = pd.DataFrame({'date': dates, 'value': values})
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        df = df.sort_values(by='date').reset_index(drop=True)
        return df

    before_df = load_openeo_json_to_df('no2_before_covid.json')
    after_df = load_openeo_json_to_df('no2_after_covid.json')
    df_raw = pd.concat([before_df, after_df])
    df_raw["value"] = df_raw["value"].interpolate(method="linear")
    df_raw = df_raw.dropna()
    df_raw = df_raw.set_index('date')
    
    # RESAMPLE KE MINGGUAN
    df_weekly = df_raw['value'].resample('W-MON').mean()
    df_weekly = df_weekly.interpolate(method="linear")
    
    df_prophet = df_weekly.reset_index().rename(columns={'date': 'ds', 'value': 'y'})
    return df_prophet

@st.cache_resource
def load_model():
    # Load model Prophet dari file JSON
    with open('prophet_model_weekly.json', 'r') as fin:
        model = model_from_json(fin.read())
    return model

# --- Main App ---
st.title("💨 Dashboard Prediksi Kualitas Udara (NO2)")
st.write("Aplikasi ini memprediksi **rata-rata mingguan** NO2 menggunakan model Prophet.")

st.warning(
    """
    **CATATAN:** Data asli sangat 'noisy' (berduri). Untuk prediksi yang akurat dan MAPE rendah, 
    model ini dilatih menggunakan **rata-rata mingguan** (`mol/m²`).
    
    Aplikasi ini akan menampilkan **Kualitas Udara Mingguan** (apakah membaik atau memburuk).
    """
)

# --- Load Data dan Model ---
try:
    historical_data = load_historical_data_weekly()
    model = load_model()
except FileNotFoundError:
    st.error("File model 'prophet_model_weekly.json' atau file data '.json' tidak ditemukan. Harap jalankan 'train_model.py' terlebih dahulu.")
    st.stop()
except Exception as e:
    st.error(f"Terjadi error saat memuat model: {e}")
    st.stop()

# --- Tampilkan Data Historis ---
st.header("Data Historis Rata-Rata Mingguan NO2")
fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(
    x=historical_data['ds'], y=historical_data['y'],
    mode='lines', name='Rata-Rata Mingguan'
))
fig_hist.update_layout(xaxis_title="Minggu", yaxis_title="Kadar NO2 (mol/m²)")
st.plotly_chart(fig_hist, use_container_width=True)


# --- Input Prediksi ---
st.header("Buat Prediksi Mingguan Baru")
weeks_to_forecast = st.slider("Pilih jumlah MINGGU untuk prediksi ke depan:", 1, 20, 4)

if st.button("Jalankan Prediksi"):
    with st.spinner("Membuat prediksi mingguan..."):
        
        # Buat 'future dataframe' untuk prediksi MINGGUAN
        future = model.make_future_dataframe(periods=weeks_to_forecast, freq='W-MON')
        
        # Buat prediksi
        forecast = model.predict(future)
        
        # Ambil data prediksi saja (setelah data historis)
        forecast_data = forecast.iloc[-weeks_to_forecast:]
        
        # --- Plot Hasil Prediksi ---
        fig_forecast = go.Figure()
        
        # Data historis (20 minggu terakhir)
        hist_plot_data = historical_data.iloc[-20:]
        fig_forecast.add_trace(go.Scatter(
            x=hist_plot_data['ds'], y=hist_plot_data['y'],
            mode='lines', name='Data Historis (Mingguan)'
        ))
        
        # Data Prediksi (yhat)
        fig_forecast.add_trace(go.Scatter(
            x=forecast_data['ds'], y=forecast_data['yhat'],
            mode='lines', name='Prediksi', line=dict(color='red')
        ))
        
        # Confidence Interval (Area)
        fig_forecast.add_trace(go.Scatter(
            x=forecast_data['ds'], y=forecast_data['yhat_upper'],
            mode='lines', name='Batas Atas (95%)', line=dict(width=0),
            showlegend=False
        ))
        fig_forecast.add_trace(go.Scatter(
            x=forecast_data['ds'], y=forecast_data['yhat_lower'],
            mode='lines', name='Batas Bawah (95%)', line=dict(width=0),
            fillcolor='rgba(255, 0, 0, 0.2)', fill='tonexty',
            showlegend=False
        ))
        
        fig_forecast.update_layout(
            title=f"Prediksi NO2 untuk {weeks_to_forecast} Minggu ke Depan",
            xaxis_title="Minggu", yaxis_title="Rata-Rata Mingguan NO2 (mol/m²)",
            hovermode="x unified"
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # --- Analisis Tren ---
        st.header("Analisis Tren Kualitas Udara")
        
        last_4_weeks_mean = historical_data['y'].iloc[-4:].mean()
        last_forecasted_value = forecast_data['yhat'].iloc[-1]
        
        delta = (last_forecasted_value - last_4_weeks_mean) / last_4_weeks_mean
        
        st.metric(
            label=f"Prediksi di akhir {weeks_to_forecast} minggu",
            value=f"{last_forecasted_value:.6f} mol/m²",
            delta=f"{delta * 100:.2f}% (vs. rata-rata 4 minggu terakhir)"
        )
        
        if delta > 0.02:
            st.error("🔴 **Tren Kualitas Udara: MEMBURUK**")
            st.write("Prediksi menunjukkan peningkatan kadar NO2 dibandingkan dengan rata-rata 4 minggu lalu.")
        elif delta < -0.02:
            st.success("🟢 **Tren Kualitas Udara: MEMBAIK**")
            st.write("Prediksi menunjukkan penurunan kadar NO2 dibandingkan dengan rata-rata 4 minggu lalu.")
        else:
            st.info("🔵 **Tren Kualitas Udara: STABIL**")
            st.write("Prediksi menunjukkan kadar NO2 relatif stabil.")

        # Tampilkan data prediksi mentah
        st.subheader("Data Prediksi:")
        st.dataframe(forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].set_index('ds'))