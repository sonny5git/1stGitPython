# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 12:32:15 2020

@author: Desktop-Sonny
"""

from pandas import DataFrame
df = DataFrame()
df['t'] = [x for x in range(10)]
print(df)

##################################
### shift time series using panda shift()

# lag order t-1
df['t - 1'] = df['t'].shift(1)
print(df)

# forecast order t+1
df['t + 1'] = df['t'].shift(-1)
print(df)

# Technically, in time series forecasting terminology the current time (t) and 
# future times (t+1, t+n) are forecast times and past observations (t-1, t-n) 
# are used to make forecasts.

# pandas shape function, 0 return rows, 1 return columns
df.shape[0]


##################################
### shift 1 time series using TimeseriesGenerator() 
import numpy as np
import pandas as pd

# define dataset
series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

n_input = 2
generator = TimeseriesGenerator(series, series, length=n_input, batch_size = 1 )

# how many samples will be prepared by the data generator for this time series.
print(f"samples {len(generator)}")

# print each samples 
for i in range(len(generator)):
    x, y = generator[i]
    print(f"{x} ==> {y}")
    
##################################
### shift n time series using TimeseriesGenerator() 
    
# from t0 to predict t + 3

# define dataset

time_step = 2

df_main = DataFrame()
df_main['t0'] = [x for x in range(10)]
df_main['t2'] = df_main['t0'].shift(-time_step)
print(df_main)

df_main.dropna(inplace=True)

n_input = 3
series_x = df_main['t0']
target_y = df_main['t2']

generator = TimeseriesGenerator(series_x, target_y, length=n_input, batch_size = 1 )

# how many samples will be prepared by the data generator for this time series.
print(f"samples {len(generator)}")

# print each samples 
for i in range(len(generator)):
    x, y = generator[i]
    print(f"{x} ==> {y}")
    
    