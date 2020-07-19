# Krishan Patel
# Stock price prediction for Amazon created using YouTube
# video by Computer Science, "Stock Price Prediction Using
# Python & Machine Learning"
# Video: https://www.youtube.com/watch?v=QIUxPv5PJOY

'''Stock Price Prediction
This program predicts the closing stock price of a corporation
based on the past 60 days, in this case Amazon, using an artificial
neural network called Long Short Term Memory (LSTM)
'''

import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

def stock_price_prediction_ltsm(stock_ticker_symbol):
    '''Predicts the stock price of a company 60 days into the future using LTSM'''
    # Get stock quote and number of rows and columns
    dataframe = web.DataReader(stock_ticker_symbol,
                               data_source='yahoo',
                               start='2012-01-01',
                               end='2020-07-17')
    # print(dataframe.shape)

    # Envisioning closing price history
    title = 'Close Price USD for ' + stock_ticker_symbol + ' ($)'
    plt.figure(figsize=(16, 8))
    plt.title(title)
    plt.plot(dataframe['Close'])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    # plt.show()

    # Create new dataframe with a only the "Close" column and
    # convert it into a numpy array
    close_dataframe = dataframe.filter(['Close'])
    close_numpy = close_dataframe.values

    # Get number of rows to train the model on (training on 80%)
    training_data_length = math.ceil(len(close_numpy) * 0.8)
    # print(training_data_length)

    # Preprocess the data (scaling data) before presented to neural network
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_numpy)
    # print(scaled_data)

    # Create the scaled training data set
    scaled_train_data = scaled_data[0 : training_data_length, :]

    # Split data into x_train and y_train data sits
    x_train = []
    y_train = []

    for i in range(60, len(scaled_train_data)):
        x_train.append(scaled_train_data[i - 60 : i, 0])
        y_train.append(scaled_train_data[i, 0])

#        if i <= 61:
#            print(x_train)
#            print(y_train)
#            print()

    # Convert the x_train and y_train variables to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data so data is 3-dimensional for LTSM network
    x_train = np.reshape(x_train, (np.size(x_train, 0), np.size(x_train, 1), 1))
    # print(x_train.shape)

    # Build the LSTM model
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, return_sequences=True, input_shape=(np.size(x_train, 1), 1)))
    lstm_model.add(LSTM(50, return_sequences=False))
    lstm_model.add(Dense(25))
    lstm_model.add(Dense(1))

stock_price_prediction_ltsm('AMZN')
