import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from datetime import datetime

# Function to fetch stock data and train the model
def fetch_and_train_model(ticker, start_date, end_date, window_size):
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))
    X, y = [], []
    for i in range(len(scaled_data) - window_size):
        feature = scaled_data[i:i + window_size]
        label = scaled_data[i + window_size][0]
        X.append(feature)
        y.append(label)
    X = np.array(X)
    y = np.array(y)

    split_index = int(len(X) * 0.65)
    X_train, X_test, y_train, y_test = X[:split_index], X[split_index:], y[:split_index], y[split_index:]

    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], 1), activation='relu', return_sequences=True))
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(LSTM(128, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    model.fit(X_train_reshaped, y_train, epochs=100, batch_size=64, verbose=2)

    return model, scaler, X_test, y_test

# Function to make predictions and plot results
def make_predictions_and_plot(model, scaler, X_test):
    X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    predictions = model.predict(X_test_reshaped)

    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    predictions_actual = scaler.inverse_transform(predictions)

    mse = mean_squared_error(y_test_actual, predictions_actual)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test_actual - predictions_actual) / y_test_actual)) * 100

    st.write(f'Mean Squared Error (MSE): {mse}')
    st.write(f'Root Mean Squared Error (RMSE): {rmse}')
    st.write(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')

    # Plot the results
    plt.figure(figsize=(12, 8))
    plt.plot(y_test_actual, label='Actual')
    plt.plot(predictions_actual, label='Predicted')
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    plt.legend()
    st.pyplot()

# Streamlit App
st.title('Stock Price Predictor')

# Sidebar for user input
ticker = st.sidebar.text_input('Enter Ticker Symbol', '^GSPC')
start_date = st.sidebar.text_input('Enter Start Date (YYYY-MM-DD)', '2010-01-01')
end_date = st.sidebar.text_input('Enter End Date (YYYY-MM-DD)', datetime.now().strftime('%Y-%m-%d'))
window_size = st.sidebar.slider('Select Window Size', min_value=1, max_value=100, value=60)

if st.sidebar.button('Run Prediction'):
    st.write(f'Predicting stock prices for {ticker} from {start_date} to {end_date} with a window size of {window_size} days...')
    
    # Fetch data and train the model
    model, scaler, X_test, y_test = fetch_and_train_model(ticker, start_date, end_date, window_size)
    
    # Make predictions and plot results
    make_predictions_and_plot(model, scaler, X_test)
