import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Streamlit app
st.title("Stock Price Prediction")

# Stock search bar
symbol = st.text_input("Enter a stock symbol (e.g., AAPL)", "AAPL")

# Download stock data for a given symbol
@st.cache
def download_stock_data(symbol):
    stock_data = yf.download(symbol, start="2010-01-01", end="2022-01-01")
    return stock_data

stock_data = download_stock_data(symbol)

# Preprocess stock data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data["Close"].values.reshape(-1, 1))
train_data = scaled_data[:int(len(scaled_data)*0.8)]
test_data = scaled_data[int(len(scaled_data)*0.8):]

# Define a function to create input sequences for the CNN model
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Create input sequences for the CNN model
seq_length = 60
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Reshape input data to match the CNN model input shape
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define the CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=(seq_length, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation="relu"))
model.add(Dense(1))

# Compile the CNN model
model.compile(loss="mean_squared_error", optimizer="adam")

# Train the CNN model
history = model.fit(X_train, y_train, epochs=3, batch_size=64, validation_data=(X_test, y_test), verbose=2)

# Predict button
if st.button("Predict"):
    # Make predictions with the CNN model
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # Reverse the scaling of the predictions
    train_predictions = scaler.inverse_transform(train_predictions)
    y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
    test_predictions = scaler.inverse_transform(test_predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate MAE, MSE, and RMSE for the test data
    mae = mean_absolute_error(y_test, test_predictions)
    mse = mean_squared_error(y_test, test_predictions)
    rmse = np.sqrt(mse)

    # Plot the training and validation loss
    st.subheader("Model Loss")
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper right")
    st.pyplot()

    # Plot the predicted and actual prices for the test data
    st.subheader("Actual vs. Predicted Prices")
    plt.plot(y_test, label="Actual")
    plt.plot(test_predictions, label="Predicted")
    plt.legend()
    st.pyplot()

    # Print the evaluation metrics
    st.subheader("Evaluation Metrics")
    st.write("Mean Absolute Error:", mae)
    st.write("Mean Squared Error:", mse)
    st.write("Root Mean Squared Error:", rmse)
