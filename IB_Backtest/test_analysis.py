import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from pylab import plt, mpl
import datetime as dt
plt.style.use('ggplot')

df = pd.read_csv(r'../stock_dfs/AAPL.csv', index_col = 0, parse_dates=True)

df['MA26'] = df['Adj Close'].rolling(26).mean()
df['MA12'] = df['Adj Close'].rolling(12).mean()

df['MACD'] = df['MA12'] - df['MA26']
df['MACD_lag1'] = df['MACD'].shift(1)
df['log_rets'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
df['Adj Close_lag1'] = df['Adj Close'].shift(1)

df['Position'] = np.where(df['MACD'] > df['MACD_lag1'], 1, -1)
df['Strategy'] = df['Position'].shift(1) * df['log_rets']

df.dropna(inplace=True)

np.exp(df[['log_rets', 'Strategy']].sum())

ax = df[['log_rets', 'Strategy']].cumsum().apply(np.exp).plot(figsize = (10,6))

df['Position'].plot(ax = ax, secondary_y = 'Position', style ='--')
ax.get_legend().set_bbox_to_anchor((0.25,0.85))

# subplot = plt.bar(df.index.values, df['MACD'])


#%% DNN w/ scikit-learn

from sklearn.neural_network import MLPClassifier

df['MACD_lead1'] = df['MACD'].shift(-1)

MACD_tuple = pd.Series(zip(df['MACD_lag1'], df['MACD'], df['MACD_lead1']))

df['MACD_tuple'] = MACD_tuple.values


df['Adj Close_lag5'] = df['Adj Close'].shift(5)
df['direction_lag5'] = np.where(df['Adj Close'] > df['Adj Close_lag5'], 1, -1)

features_array = ['MACD_tuple','Adj Close_lag5']
model = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1)

model.fit(df[features_array], df['direction_lag5'])

df['pos_dnn_sk'] = model.predict(data['Adj Close'])

df['strat_dnn_sk'] = df['pos_dnn_sk'] * df['log_rets']

data[['returns', 'strat_dnn_sk']].cumsum().apply(np.exp).plot(figsize = (10, 6))