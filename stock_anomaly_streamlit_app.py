import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
from sklearn.ensemble import IsolationForest
from prophet import Prophet

st.set_page_config(page_title="Financial Anomaly Detection", layout="wide")

st.title("📈 Financial Time-Series Anomaly Detection Tool")
st.write("Upload your Yahoo Finance data (`.xlsx` file with a `Close*` column). The app will detect price anomalies and show forecasts.")

# File upload
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("### Raw Data", df.head())

    # Preprocess
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # Indicator Calculation
    df['SMA_20'] = ta.trend.sma_indicator(df['Close*'], window=20)
    df['EMA_20'] = ta.trend.ema_indicator(df['Close*'], window=20)
    df['RSI_14'] = ta.momentum.rsi(df['Close*'], window=14)
    boll = ta.volatility.BollingerBands(close=df['Close*'], window=20, window_dev=2)
    df['BB_High'] = boll.bollinger_hband()
    df['BB_Low'] = boll.bollinger_lband()

    features = df[['Close*', 'SMA_20', 'EMA_20', 'RSI_14', 'BB_High', 'BB_Low']].dropna()
    X = features.values

    # Anomaly Detection
    iso_forest = IsolationForest(contamination=0.02, random_state=42)
    anomaly_labels = iso_forest.fit_predict(X)
    features['Anomaly'] = anomaly_labels
    features['Date'] = df.loc[features.index, 'Date'].values
    features['Close*'] = df.loc[features.index, 'Close*'].values

    # Prophet Forecasting
    prophet_df = df[['Date', 'Close*']].rename(columns={'Date': 'ds', 'Close*': 'y'}).dropna()
    model = Prophet(daily_seasonality=True)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    result = pd.merge(prophet_df, forecast[['ds', 'yhat']], on='ds', how='left')
    result['Deviation'] = np.abs(result['y'] - result['yhat'])

    # --- Plot 1: Anomaly chart ---
    fig1, ax1 = plt.subplots(figsize=(12,5))
    ax1.plot(features['Date'], features['Close*'], label='Closing Price', color='blue')
    ax1.scatter(features.loc[features['Anomaly']==-1, 'Date'],
                features.loc[features['Anomaly']==-1, 'Close*'],
                color='red', label='Anomaly', marker='x', s=80)
    ax1.set_title('Stock Price with Detected Anomalies')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Closing Price')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig1)

    # --- Plot 2: Prophet Forecast ---
    fig2 = model.plot(forecast)
    ax2 = fig2.gca()
    ax2.set_title('Stock Price Forecast (Prophet)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Predicted Closing Price')
    ax2.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig2)

    # --- Plot 3: Deviation ---
    fig3, ax3 = plt.subplots(figsize=(12,5))
    ax3.plot(result['ds'], result['Deviation'], color='purple', label='Absolute Deviation')
    ax3.set_title('Deviation Between Actual and Forecasted Prices')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Absolute Deviation')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig3)

    # Show anomaly table
    anomalies = features[features['Anomaly']==-1][['Date','Close*']]
    anomalies = anomalies.sort_values('Date').tail(10)
    anomalies['Date'] = anomalies['Date'].dt.strftime('%Y-%m-%d')
    st.write("### Recent Detected Anomalies", anomalies)

    st.success(f"Total anomalies detected: {(anomaly_labels == -1).sum()}")
else:
    st.info("Please upload a Yahoo Finance Excel file to begin.")

