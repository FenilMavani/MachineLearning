import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

def main():
    st.title('Stock Market Prediction')
    st.write('Enter the stock symbol and select the prediction period:')
    
    symbol = st.text_input('Stock Symbol (e.g., AAPL for Apple):')
    period = st.selectbox('Prediction Period', ['1d', '5d', '1mo', '3mo'])
    
    if st.button('Predict'):
        st.write('Fetching stock data...')
        df = yf.download(symbol, period=period)
        df = df[['Close']]
        
        st.write('Preprocessing data...')
        dataset = df.values
        dataset = dataset.astype('float32')
        
        train_size = int(len(dataset) * 0.8)
        train_data, test_data = dataset[:train_size], dataset[train_size:]
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_data = scaler.fit_transform(train_data)
        
        def create_dataset(data, lookback):
            dataX, dataY = [], []
            for i in range(len(data) - lookback - 1):
                a = data[i:(i + lookback), 0]
                dataX.append(a)
                dataY.append(data[i + lookback, 0])
            return np.array(dataX), np.array(dataY)
        
        lookback = 20
        trainX, trainY = create_dataset(train_data, lookback)
        
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        
        st.write('Building and training the model...')
        model = Sequential()
        model.add(LSTM(50, input_shape=(1, lookback)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
        
        st.write('Predicting stock prices...')
        # Previous code...

testX, testY = create_dataset(test_data, lookback)

if len(testX.shape) < 3:
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

predicted_prices = model.predict(testX)

# Subsequent code...

        predicted_prices = scaler.inverse_transform(predicted_prices)
        
        st.write('Plotting the results...')
        df['Date'] = df.index
        df_train = df[:train_size]
        df_test = df[train_size + lookback + 1:]
        df_test['Predicted'] = predicted_prices
        
        st.line_chart(df[['Close', 'Predicted']])
        st.write(df_test)

if __name__ == '__main__':
    main()
