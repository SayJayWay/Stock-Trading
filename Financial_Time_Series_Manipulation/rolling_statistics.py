import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

data = pd.read_csv('./stock_dfs/AAPL.csv').dropna()

window = 20

data['min'] = data['Adj Close'].rolling(window = window).min()
data['max'] = data['Adj Close'].rolling(window = window).max()
data['mean'] = data['Adj Close'].rolling(window = window).mean()
data['std'] = data['Adj Close'].rolling(window = window).std()
data['median'] = data['Adj Close'].rolling(window = window).median()
data['ewma'] = data['Adj Close'].ewm(halflife = 0.5, min_periods = window).mean()


print(data.columns)
