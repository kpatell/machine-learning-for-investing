# Krishan Patel
# Stock price prediction for Amazon created using YouTube
# video by Computer Science, "Stock Price Prediction Using
# Python & Machine Learning"
# Video: https://www.youtube.com/watch?v=QIUxPv5PJOY

"""Stock Price Prediction
This program predicts the closing stock price of a corporation
based on the past 60 days, in this case Amazon, using an artificial
neural network called Long Short Term Memory (LSTM)
"""

import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Get stock quote and number of rows and columns
DATAFRAME = web.DataReader('AMZN',
                           data_source='yahoo',
                           start='2012-01-01',
                           end='2020-07-17')
# DATAFRAME.shape

# Envisioning closing price history
plt.figure(figsize=(16, 8))
plt.title('Close Price History')
plt.plot(DATAFRAME['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()
