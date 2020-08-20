import pandas as pd
import numpy as np

data = pd.read_csv('../stock_dfs/AAPL.csv', index_col = 0).dropna()

window = 20

# Rolling statistics for min, max, mean ,std, median
data['min'] = data['Adj Close'].rolling(window = window).min()
data['max'] = data['Adj Close'].rolling(window = window).max()
data['mean'] = data['Adj Close'].rolling(window = window).mean()
data['std'] = data['Adj Close'].rolling(window = window).std()
data['median'] = data['Adj Close'].rolling(window = window).median()
data['ewma'] = data['Adj Close'].ewm(halflife = 0.5, min_periods = window).mean()

ax = data[['min', 'mean', 'max']].iloc[-200:].plot(
        figsize = (10,6), style=['g--', 'r--', 'g--'], lw=0.8)
#data['Adj Close'].iloc[-200:].plot(ax = ax, lw= 2.0)

# 50SMA vs 200MA
data['50SMA'] = data['Adj Close'].rolling(window = 50).mean()
data['200SMA'] = data['Adj Close'].rolling(window = 200).mean()
data[['Adj Close', '50SMA', '200SMA']].plot(figsize = (10,6))

# Clean up graph and set positions
data.dropna(inplace = True)
data['positions'] = np.where(data['50SMA'] > data['200SMA'], 1, -1)

ax = data[['Adj Close', '50SMA', '200SMA', 'positions']].plot(figsize = (10,6), secondary_y = 'positions')
ax.get_legend().set_bbox_to_anchor((0.25, 0.85))