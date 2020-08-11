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
                               end='2019-12-17')
    # print(dataframe)
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
        x_train.append(scaled_train_data[i-60 : i, 0])
        y_train.append(scaled_train_data[i, 0])

#        if i <= 61:
#            print(x_train)
#            print(y_train)
#            print()

    # Convert the x_train and y_train variables to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data so data is 3-dimensional for LTSM network
    x_train = np.reshape(x_train, (np.size(x_train, 0), np.size(x_train, 1), 1))
    # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # print(x_train.shape)

    # Build the LSTM model
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(np.size(x_train, 1), 1)))
    # lstm_model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
    lstm_model.add(LSTM(units=50, return_sequences=False))
    lstm_model.add(Dense(units=25))
    lstm_model.add(Dense(units=1))

    # Compiling the model
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    lstm_model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Create the testing data set
    # Create new array that contains the scaled values from index 60 less than max (60 days)
    test_data = scaled_data[training_data_length-60:, :]

    # Create the data sets x_test and y_test (not scaled yet)
    x_test_data = []

    # Get all of the rows from index 60 less to the rest and all
    # of the columns (in this case, it's only "Close"). Number
    # of rows to get would just be:
    # training_data_length - num_future_days = # rows of data
    y_test_data = close_numpy[training_data_length :, :]

    for i in range(60, len(test_data)):
        x_test_data.append(test_data[i-60 : i, 0])

    # Convert x_test_data into a numpy array to use in LSTM model
    x_test_data = np.array(x_test_data)

    # Reshape the data to make it three-dimensional to be accepted by the LSTM
    x_test_data = np.reshape(x_test_data, (np.size(x_test_data, 0), np.size(x_test_data, 1), 1))
    # x_test_data = np.reshape(x_test_data, (x_test_data.shape[0], x_test_data.shape[1], 1))
    # print(x_test_data.shape)

    # Getting the models predicted price values
    # These should be the same as the y_test_data set because
    # we are checking to see if the predictions are the same
    # as the last 60 days of the stock
    test_predictions = lstm_model.predict(x_test_data)
    test_predictions = scaler.inverse_transform(test_predictions) # Undoes the scaling

    # Get the root mean squared error (RMSE) - lower values indicate better fit
    # to see how well the model performs.
    rmse = np.sqrt(np.mean((test_predictions - y_test_data)**2))
    # print(rmse)

    # Plot the data
    train = close_dataframe[:training_data_length]
    valid = close_dataframe[training_data_length:]
    valid['Predictions'] = test_predictions

    # Visualize the data
    model_title = 'Model Close Price USD for ' + stock_ticker_symbol + ' ($)'
    plt.figure(figsize=(16, 8))
    plt.title(model_title)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close']) # closing prices before predictions
    plt.plot(valid['Close']) # the actual closing prices
    plt.plot(valid[['Close', 'Predictions']]) # the predicted closing prices
    plt.legend(['Train', 'Values', 'Predictions'], loc='lower right')
    plt.show()

    # Showing the valid and predicted prices
    print(valid)

stock_price_prediction_ltsm('AMZN')
