# Example online marketing
# 
# Tracking User 1 features/labels: page views, conversion, bounce rates from 2019 - 2020
# Tracking User 2 features/labels: page views, conversion, bounce rates from 2019 - 2020
# User 1 --> User n : samples : N = 2
# page views, conversion, bounce rates: D = 3
# from 2019 - 2020: page views, conversion and bounce rates recored daily, T = 365
#
# Example 1 stock price
# 
# D = 1 (the price)
# Using 10 window size of stock price to predict next value 
# T = 10
# N = number of windows in time series
# If we have sequence of 100 stock prices, how many windows of size 10
# 100 - 10 + 1 = 91
# 
# Fomular: sequency of length L, window size T, N = L - T + 1
#
# Example 500 stock prices
# D = 500 
# Time window T = 10 
#  
# All ML librarires in Python conform  N x T x D standard 
# N first
# D last (D is the number of features)

## time series of length 10 
# predict next value using past 3 values 
# D = 1 
# T = 3 
# Input = N x 3
# Output = N
# N = L - T + 1 = 10 - 3 + 1 = 8
# But we don't have predict value 11 to fit into the model,
# the python model requires input and ouput 
# So N = L - T + 1 - Pt (predict time step) = 10 - 3 + 1 - 1 = 7 

series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

n_input = 3
generator = TimeseriesGenerator(series, series, length=n_input, batch_size = 1 )

# how many samples will be prepared by the data generator for this time series.
print(f"samples {len(generator)}")

# print each samples 
for i in range(len(generator)):
    x, y = generator[i]
    print(f"{x} ==> {y}")
