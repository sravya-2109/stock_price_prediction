#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import date
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense
import tensorflow as tf
import random
import os
from ta.momentum import RSIIndicator
from ta.trend import MACD

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
set_seed(41)  # You can use any number, but be consistent

# Set page configuration
st.set_page_config(layout="wide")
st.title("📈 Stock Analysis Dashboard")

# Define available tickers
tickers = ["AAPL", "MSFT", "GOOGL"]

# Create a 2x2 grid layout
top_left, top_right = st.columns(2)
bottom_left, bottom_right = st.columns(2)

# Top-Left: Stock Selection and Date Input
today = date.today()
with top_left:
    st.subheader("Stock Selection & Prediction Date")
    selected_ticker = st.selectbox("Select a stock", tickers)
    input_date_str = st.text_input("Enter current date (YYYY-MM-DD):")

    # Download stock data
    stock_data = yf.download(selected_ticker, start="2020-01-01", end = today)
    sector_data = yf.download("XLK", start="2020-01-01", end = today)
    # Reset index to move 'Date' from index to a column
    stock_data.reset_index(inplace=True)
    sector_data.reset_index(inplace=True)
    stock_data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in stock_data.columns.values]
    close_col = f"Close_{selected_ticker}"

    # Predict Future Price
    if input_date_str:
        try:
            input_date = datetime.strptime(input_date_str, "%Y-%m-%d")
            last_date = stock_data['Date_'].iloc[-1]
            if input_date <= last_date:
                st.warning("Please enter a future date.")
            else:
                # Prepare data for prediction
                close_prices = stock_data[close_col].values.reshape(-1, 1)
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(close_prices)
                sequence_length = 60
                X, y = [], []
                for i in range(sequence_length, len(scaled_data)):
                    X.append(scaled_data[i-sequence_length:i, 0])
                    y.append(scaled_data[i, 0])
                X = np.array(X)
                y = np.array(y)
                X = np.reshape(X, (X.shape[0], X.shape[1], 1))

                # Build and train GRU model
                model = Sequential()
                model.add(GRU(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
                model.add(Dropout(0.2))
                model.add(GRU(units=50, return_sequences=False))
                model.add(Dropout(0.2))
                model.add(Dense(units=1))
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(X, y, epochs=10, batch_size=32, verbose=0)

                # Predict future price
                today_price = close_prices[-1][0]
                st.success(f"Price on {input_date.date()}: ${today_price:.2f}")
                n_days = st.selectbox("Select prediction window (days):", [7, 15, 30])
                last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)

                predicted_prices = []
                prediction_dates = []

                current_date = datetime.now()

                for i in range(n_days):
                    next_pred_scaled = model.predict(last_sequence)[0][0]
                    predicted_price = scaler.inverse_transform([[next_pred_scaled]])[0][0]

                    predicted_prices.append(predicted_price)
                    prediction_dates.append(current_date + timedelta(days=i+1))

                    # Prepare new input sequence for next prediction
                    new_sequence = np.append(last_sequence[0, 1:, 0], next_pred_scaled)  # drop first, append new
                    last_sequence = new_sequence.reshape(1, sequence_length, 1)

                # Convert to DataFrame for visualization
                forecast_df = pd.DataFrame({
                    'Date': prediction_dates,
                    'Predicted Close Price': predicted_prices
                })
                # Today's actual closing price
                today_close_price = close_prices[-1][0]

                # Calculate mean predicted price
                mean_predicted_price = np.mean(predicted_prices)

                # Recommendation logic
                if mean_predicted_price > today_close_price :
                    recommendation = "We recommend to Buy the stock"
                else:
                    recommendation = "We recommend not to buy the stock"

# Display results
                st.success(f"Mean Predicted Price (Next {n_days} days): ${mean_predicted_price:.2f}")
                st.markdown(f"### 📌 Recommendation: **{recommendation}**")

        except ValueError:
            st.error("Invalid date format. Please use YYYY-MM-DD.")



# Top-Right: Date vs. Close Prices
# Reset index to move 'Date' from index to a column
stock_data.reset_index(inplace=True)

# Construct the column name for the selected ticker's closing price
with top_right:
    st.subheader("Price Trends")
    if close_col in stock_data.columns:
    # Plot the closing prices
        fig = px.line(stock_data, x='Date_', y=close_col, title=f'{selected_ticker} Closing Prices Over Time')
        st.plotly_chart(fig, use_container_width=True)

# Bottom-Left: Moving Averages
with bottom_left:
    st.subheader("Moving Averages")

    # Construct the column name for the selected ticker's closing price
    close_col = f"Close_{selected_ticker}"

    # Check if the constructed column exists in the DataFrame
    if close_col in stock_data.columns:
        # Create a copy of the DataFrame to avoid modifying the original data
        stock_data = stock_data[['Date_', close_col]].copy()
        stock_data.rename(columns={close_col: 'Close'}, inplace=True)

        # Calculate moving averages
        stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
        stock_data['MA200'] = stock_data['Close'].rolling(window=200).mean()

        # Plot the closing price along with moving averages
        fig_ma = px.line(
            stock_data,
            x='Date_',
            y=['Close', 'MA50', 'MA200'],
            labels={'value': 'Price', 'variable': 'Legend'},
            title=f'{selected_ticker} Closing Price with 50 & 200 Day Moving Averages'
        )
        st.plotly_chart(fig_ma, use_container_width=True)
with bottom_right:
    start_date = "2020-01-01"
    end_date = today

    data = yf.download(selected_ticker, start=start_date, end=end_date, auto_adjust=False)
    xlk = yf.download("XLK", start=start_date, end=end_date, auto_adjust=False)

    data['Daily Return'] = data['Adj Close'].pct_change()
    xlk['Daily Return'] = xlk['Adj Close'].pct_change()

    data['Cumulative Return'] = (1 + data['Daily Return']).cumprod()
    xlk['Cumulative Return'] = (1 + xlk['Daily Return']).cumprod()
    fig = go.Figure()


    fig.add_trace(go.Scatter(x=data.index, y=data['Cumulative Return'],
                             mode='lines', name=selected_ticker,
                             line=dict(color='blue')))

    # Add XLK cumulative return trace
    fig.add_trace(go.Scatter(x=xlk.index, y=xlk['Cumulative Return'],
                             mode='lines', name='XLK (Tech Sector)',
                             line=dict(color='green')))

    fig.update_layout(title=f'Cumulative Returns: {selected_ticker} vs. Technology Sector (2020–2024)',
                      xaxis_title='Date',
                      yaxis_title='Cumulative Return',
                      hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

rsi_indicator = RSIIndicator(close=stock_data['Close'], window=14)
stock_data['RSI'] = rsi_indicator.rsi()

# Calculate MACD
macd_indicator = MACD(close=stock_data['Close'])
stock_data['MACD'] = macd_indicator.macd()
stock_data['Signal Line'] = macd_indicator.macd_signal()
st.subheader("Relative Strength Index (RSI)")
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=data.index, y=stock_data['RSI'], name='RSI'))
fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
fig_rsi.update_layout(title='RSI Indicator', xaxis_title='Date', yaxis_title='RSI')
st.plotly_chart(fig_rsi, use_container_width=True)

# Plot MACD
st.subheader("Moving Average Convergence Divergence (MACD)")
fig_macd = go.Figure()
fig_macd.add_trace(go.Scatter(x=data.index, y=stock_data['MACD'], name='MACD'))
fig_macd.add_trace(go.Scatter(x=data.index, y=stock_data['Signal Line'], name='Signal Line'))
fig_macd.update_layout(title='MACD Indicator', xaxis_title='Date', yaxis_title='MACD')
st.plotly_chart(fig_macd, use_container_width=True)


# In[ ]:




