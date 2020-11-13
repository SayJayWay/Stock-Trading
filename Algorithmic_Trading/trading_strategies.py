# Trading strategies covered
# - Simple MA
# - Random Walk Hypothesis
# - Linear OLS Regression
# - Clustering
# - Frequency Approach
# - Classification
# - Deep Neural Networks
#%%
# =============================================================================
# Simple MA
# =============================================================================

import numpy as np
import pandas as pd
import datetime as dt
from pylab import mpl, plt

symbol = 'Adj Close'
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

raw = pd.read_csv('../stock_dfs/AAPL.csv', index_col = 0)

data = (pd.DataFrame(raw[symbol]).dropna())

SMA1 = 42
SMA2 = 252

data['SMA1'] = data[symbol].rolling(SMA1).mean()
data['SMA2'] = data[symbol].rolling(SMA2).mean()

data.plot(figsize = (10, 6))

# if 42SMA > 252SMA -> long (1), else short (-1)

data.dropna(inplace = True)

data['Position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)

ax = data.plot(secondary_y = 'Position', figsize = (10, 6))
ax.get_legend().set_bbox_to_anchor((0.25, 0.85))

# Vectorized Backtesting
# Assumes 0 transaction costs (fees, bid-ask spread, etc).  This is justifiable
# for strategies that make few trades over multiple years.  It also assumes
# that all trades take place at the end of the day.  A more realistic approach
# would take these and other elements into account

# Calculating performance
data['Returns'] = np.log(data[symbol] / data[symbol].shift(1)) # log returns
# Multiplies position values, shifted by one day, by the log returns; the shift
# is required to avoid foresight bias
data['Strategy'] = data['Position'].shift(1) * data['Returns']
data.dropna(inplace = True)

# Sums up the log returns for strategy and the benchmark investment and 
# calculates the exp value to arrive at absolute performance
np.exp(data[['Returns', 'Strategy']].sum())
# Annualized volatility for strategy and benchmark investment
data[['Returns', 'Strategy']].std() * 252 ** 0.5

ax = data[['Returns', 'Strategy']].cumsum().apply(np.exp).plot(figsize = (10, 6))
data['Position'].plot(ax = ax, secondary_y = 'Position', style = '--')
ax.get_legend().set_bbox_to_anchor((0.25, 0.85))

print(np.exp(data[['Returns', 'Strategy']].sum()))

# Optimization
# Are 42SMA and 252SMA the right ones?  We can use vectorized backtesting and
# try all the parameter combinations, record the results and rank them.

from itertools import product

sma1 = range(20, 61, 4)
sma2 = range(180, 281, 10)

results = pd.DataFrame()
for SMA1, SMA2 in product(sma1, sma2): # Combines all values for SMA1 against SMA2
    data = pd.DataFrame(raw[symbol])
    data.dropna(inplace = True)
    data['Returns'] = np.log(data[symbol] / data[symbol].shift(1))
    data['SMA1'] = data[symbol].rolling(SMA1).mean()
    data['SMA2'] = data[symbol].rolling(SMA2).mean()
    data.dropna(inplace = True)
    data['Position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)
    data['Strategy'] = data['Position'].shift(1) * data['Returns']
    data.dropna(inplace = True)
    perf = np.exp(data[['Returns', 'Strategy']].sum())
    results = results.append(pd.DataFrame(
            {'SMA1' : SMA1, 'SMA2' : SMA2,
             'MARKET' : perf['Returns'],
             'STRATEGY' : perf['Strategy'],
             'OUT' : perf['Strategy'] - perf['Returns']},
             index = [0]), ignore_index = True) # Records vectorized backtest 
    
# Overview of results
results.info()
results.sort_values('OUT', ascending = False).head()

# Can see that the best results are with 28SMA and 180SMA, whoever, it still
# underperforms the market.  This result is heavily dependent on the dataset 
# used and is pront to overfitting.  Better to have a test/train set

#%%
# Random Walk Hypothesis (RWH)
# States that strategies based on a single financial time series, that create
# predictive approaches should not yield any outperformance at all.  It postulates
# that prices in financial markets follow a random walk, or an arithmetic 
# Brownian motion without drift, where the expected value at any point of time
# equals its value today.

# In other words, the best predictor for tomorrow's price, in least-squares sense
# is today's price if RWH applies.  RWH is consistent with efficient markets
# hypothesis (EMH), which essentially states that market prices reflect 'all 
# available information'. Different degrees of efficiency are generally distinguished,
# "weak", "semi-strong", "strong", which defines more specifically what 'all
# available infoamtion' entails.  

# Example: A financial time series of historical market prices is used for which
# a large number of "lagged" versions are created (eg. 5).  OLS regression can
# be used to predict the market prices based on the lagged versions.  Basic idea
# is that the market prices from yesterday and four more days back can be used
# to predict today's market price

symbol = 'SPY'

raw = pd.read_csv('../stock_dfs/{}.csv'.format(symbol), index_col = 0)

data[symbol] = pd.DataFrame(raw['Adj Close'])

lags = 5
cols = []

for lag in range(1, lags + 1):
    col = 'lag_{}'.format(lag) # Defines column for current lag value
    data[col] = data[symbol].shift(lag) # Creates lagged version
    cols.append(col) # Collects column names for later reference

data.dropna(inplace = True)

# lag_1 is generally what is used to come up with the prediction value as it is 
# closest to present price.

#%%
# =============================================================================
# Linear OLS Regression
# =============================================================================
# Will use two features: lag_1 which represents log returns of the financial
# time series lagged by one day and lag_2 which is by two days.  Log returns
# (in contrast to prices) are 'stationary', which is often a necessary condition
# for the application of statistical and ML algorithms.

# Basic idea of using lagged log returns is that they might be informative in
# predicting future returns.  Ex, one might hypothesize that after two downward
# movements, an upward movement is more likely (mean reversion) or, contrarily,
# another downward movement is more likely (momentum/trend).  Regression
# techniques allow for formalization of such informal reasonings.

raw = pd.read_csv('../stock_dfs/EUR=X.csv', index_col = 0)
del data
symbol = 'EUR='
data = pd.DataFrame(raw['Adj Close'])

data['returns'] = np.log(data / data.shift(1))
data.dropna(inplace = True)

data['direction'] = np.sign(data['returns']).astype(int)
data['returns'].hist(bins = 35, figsize = (10, 6))

lags = 2

def create_lags(data):
    global cols
    cols = []
    for lag in range(1, lags + 1):
        col = 'lag_{}'.format(lag)
        data[col] = data['returns'].shift(lag)
        cols.append(col)
        
create_lags(data)

data.dropna(inplace = True)
data.plot.scatter(x = 'lag_1', y = 'lag_2', c = 'returns', cmap = 'coolwarm',
                  figsize = (10, 6), colorbar = True)
plt.axvline(0, c = 'r', ls = '--')
plt.axhline(0, c = 'r', ls = '--')
plt.title('Scatter plot based on features and labels data')

# Linear OLS can now be implemented to learn about any potential linear
# relationships, to predict market movement based on features and to backtest
# such predictions.  Two basic approaches are available: using log returns or only
# the direction data as the dependent variable.

from sklearn.linear_model import LinearRegression

model = LinearRegression()

# Regression on log returns directly
data['pos_ols_1'] = model.fit(data[cols], data['returns']).predict(data[cols])
# Regression on direction data (which is of primary interest)
data['pos_ols_2'] = model.fit(data[cols], data['direction']).predict(data[cols])

# Real-valued predictions are transformed to directional values (+1, -1)
data[['pos_ols_1', 'pos_ols_2']] = np.where(data[['pos_ols_1', 'pos_ols_2']] > 0, 1, -1)

# Two approaches yield different directional predictions in general
data['pos_ols_1'].value_counts()
#-1    1155
# 1      65
data['pos_ols_2'].value_counts()
#-1    707
# 1    513

# However, both lead to relatively large number of trades over time
(data['pos_ols_1'].diff() != 0).sum() # 111
(data['pos_ols_2'].diff() != 0).sum() # 739

data['strat_ols_1'] = data['pos_ols_1'] * data['returns']
data['strat_ols_2'] = data['pos_ols_2'] * data['returns']

data[['returns', 'strat_ols_1', 'strat_ols_2']].sum().apply(np.exp)

# Number of correct and false predictions by the strategies
(data['direction'] == data['pos_ols_1']).value_counts()
(data['direction'] == data['pos_ols_2']).value_counts()

data[['returns', 'strat_ols_1', 'strat_ols_2']].cumsum().apply(np.exp).plot(figsize = (10, 6))
plt.title('Performance of EUR/USD and regression-based strategies over time')

# Assuming zero transaction costs, and same data set for training and testing,
# Can see that both of these outperform benchmark passive investment (returns),

#%% 
# =============================================================================
# Clustering
# =============================================================================
# k-means clustering on financial time series data to automatically come up w/
# clusters that are used to formulate a trading strategies.  Idea is that the
# algorithm identifies 2 clusters of feature values that predict either an
# upward or downward movement

from sklearn.cluster import KMeans

model = KMeans(n_clusters = 2, random_state = 0) # Two clusters chosen for algo

model.fit(data[cols])

KMeans(algorithm = 'auto', copy_x = True, init='k-means++', max_iter = 300,
       n_clusters = 2, n_init = 10, n_jobs = None, precompute_distances = 'auto',
       random_state = 0, tol = 0.0001, verbose = 0)

data['pos_clus'] = model.predict(data[cols])
# Given cluster values, position is chosen
data['pos_clus'] = np.where(data['pos_clus'] == 1, -1, 1)

data['pos_clus'].values

plt.figure(figsize = (10, 6))
plt.scatter(data[cols].iloc[:, 0], data[cols].iloc[:, 1],
            c = data['pos_clus'], cmap = 'coolwarm')
plt.title('Two clusters as determined by k-means algorithm')

# This approach is quite arbitrary in this context as we didn't really specify
# what the algorithm should look for, however, it seems to do a bit better than
# benchmark.  We can see that the hit ratio - i.e. the number of correct 
# predictions is less than 50%

data['strat_clus'] = data['pos_clus'] * data['returns']

data[['returns', 'strat_clus']].sum().apply(np.exp)

(data['direction'] == data['pos_clus']).value_counts()

data[['returns', 'strat_clus']].cumsum().apply(np.exp).plot(figsize = (10, 6))

#%%
# =============================================================================
# Frequency Approach
# =============================================================================
# One could transform the two real-valued features to binary ones and assess
# the probability of an upward and downward movement, from historical 
# observations of such movements, given the four possible combinations for the
# two binary features ((0, 0), (0, 1), (1, 0), (1, 1))

def create_bins(data, bins = [0]):
    global cols_bin
    cols_bin = []
    for col in cols:
        col_bin = col + '_bin'
        data[col_bin] = np.digitize(data[col], bins = bins)
        cols_bin.append(col_bin)
        
create_bins(data)

data[cols_bin + ['direction']].head()

#            lag_1_bin  lag_2_bin  direction
#Date                                       
#2011-01-06          1          1          1
#2011-01-07          1          1          1
#2011-01-10          1          1          1
#2011-01-11          1          1         -1
#2011-01-12          0          1         -1

# Shows frequency of possible movements conditional on the feature value of combos
grouped = data.groupby(cols_bin + ['direction'])
grouped.size()

#lag_1_bin  lag_2_bin  direction
#0          0          -1           302
#                       0             3
#                       1           303
#           1          -1           307
#                       0             3
#                       1           341
#1          0          -1           329
#                       0             2
#                       1           319
#           1          -1           322
#                       0             6
#                       1           285

# Transforms df object ot have frequencies in columns
res = grouped['direction'].size().unstack(fill_value = 0)

# Highlights the highest-frequency value per feature value combination
def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]

res.style.apply(highlight_max, axis = 1)

# Given the frequency data, 3 feature value combinations hint at a downward
# movement while one lets an upward movement seem more likely.  This translates
# into a trading strategy

# Translates findings given frequencies into a trading strategy
data['pos_freq'] = np.where(data[cols_bin].sum(axis = 1) == 2, -1, 1)

(data['direction'] == data['pos_freq']).value_counts()
#True     1285
#False    1237
#dtype: int64

data['strat_freq'] = data['pos_freq'] * data['returns']
data[['returns', 'strat_freq']].sum().apply(np.exp)
#returns       1.126577
#strat_freq    1.681175
#dtype: float64

data[['returns', 'strat_freq']].cumsum().apply(np.exp).plot(figsize = (10, 6))
plt.title('Performance of EUR/USD and frequency-based trading strategy over time')

#%%
# =============================================================================
# Classification
# =============================================================================
# Application of logistic regression, Gaussian Naive Bayes, and support vector
# machine appaorches are straightforward to predicting direction of price
# movements in financial markets

# Two binary features
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

C = 1

models = {
        'log_reg' : linear_model.LogisticRegression(C = C),
        'gauss_nb': GaussianNB(),
        'svm': SVC(C = C)
        }

def fit_models(data):
    mfit = {model: models[model].fit(data[cols_bin],
            data['direction']) for model in models.keys()}
    
fit_models(data) # Function fits all models

# Function derives all position values from fitted models
def derive_positions(data):
    for model in models.keys():
        data['pos_' + model] = models[model].predict(data[cols_bin])
        
derive_positions(data)

# Vectorized backtesting
# Function that evaluates all resulting trading strategies
def evaluate_strats(data):
    global sel
    sel = []
    for model in models.keys():
        col = 'strat_' + model
        data[col] = data['pos_' + model] * data['returns']
        sel.append(col)
    sel.insert(0, 'returns')
    
evaluate_strats(data)

sel.insert(1, 'strat_freq')
data[sel].sum().apply(np.exp)# Some strategies might show exact same performance
#
#returns           1.110278
#strat_freq        1.749123
#strat_log_reg     1.385577
#strat_gauss_nb    1.647361
#strat_svm         5.052729
#dtype: float64

data[sel].cumsum().apply(np.exp).plot(figsize = (10, 6)) 
plt.title('Performance EUR/SD and classification-based trading strategies (2 binary lags)')

#%%
# Five binary features
# Can try to improve performance with 5 binary lags instead of 2.
data[symbol] = pd.DataFrame(raw['Adj Close'])

data['returns'] = np.log(data[symbol] / data[symbol].shift(1))

data['direction'] = np.sign(data['returns'])

lags = 5 # Five lags now
create_lags(data)
data.dropna(inplace = True)

create_bins(data) # Real-valued features data transformed to binary data

data[cols_bin].head()

data.dropna(inplace = True)
fit_models(data)
derive_positions(data)
evaluate_strats(data)
data[sel].sum().apply(np.exp)

# We can see that the SVM-based strategy is significantly improved, however,
# the LR- and GNB-based strategies have worsened
data[sel].cumsum().apply(np.exp).plot(figsize = (10, 6))
plt.title('Performance EUR/USD and classification-based trading strategies (5 binary legs)')

#%% Five Digitized Features

mu = data['returns'].mean() # Mean log return..
v = data['returns'].std() # Std ..

bins = [mu - v, mu, mu + v] #..used to digitize bins

create_bins(data, bins)

data[cols_bin].head()

fit_models(data)
derive_positions(data)

evaluate_strats(data)
data[sel].sum().apply(np.exp)

data[sel].cumsum().apply(np.exp).plot(figsize = (10, 6))

#%% Sequential Train-Test Split
# Idea here is to simulate the situation where only data up to a certain point
# in time is available on which to train an ML algo.  During live trading,
# the algo is then faced with data it has not seen before.  In this particular
# example, all the classification algos outperform - under the simplified 
# assumptions from before

split = int(len(data) * 0.5)

train = data.iloc[:split].copy() # Training data

fit_models(train)

test = data.iloc[split:].copy()

derive_positions(test)

evaluate_strats(test)

test[sel].sum().apply(np.exp)

test[sel].cumsum().apply(np.exp).plot(figsize = (10, 6))
plt.title('Performance EUR/USD and classification-based trading strategies (sequential train-test split)')

#%% Randomized Train-Test Split
# The algo is trained and tested on binary or digitized features data.  Idea is
# that the feature value patterns allow a prediction of futur market movements
# with a better hit ratio than 50%.  Implicitly, it is assumed that the pattern's
# predictive power persists over time.  In that sense, it should not make too much
# of a difference on which part of the data the algorithm is trained on and which
# part it is tested, which implies that one can break up the temporal sequence
# of the data for training and testing

# Typical way to do randomized train-test split is out-of-sample.  

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size = 0.5, shuffle = True, random_state = 100)

train = train.copy().sort_index() # copied and brought back in temporal order

test = test.copy().sort_index() # copied and brought back in temporal order

fit_models(train)

derive_positions(test)

evaluate_strats(test)

test[sel].sum().apply(np.exp)

test[sel].cumsum().apply(np.exp).plot(figsize = (10, 6))
plt.title('Performance of EUR/USD and classification-based trading strategies (randomized train-test split)')

#%%
# =============================================================================
# Deep Neural Networks
# =============================================================================
# This example applies MLPClassifier algorithm from scikit-learn.  Algo is
# trained and tested on the whole data set, using the digitize features.  It
# achieves unrealistically good performance which suggests overfitting

from sklearn.neural_network import MLPClassifier

model = MLPClassifier(solver = 'lbfgs', alpha=1e-5, hidden_layer_sizes = 2 * [250],
                      random_state = 1)

model.fit(data[cols_bin], data['direction'])# %time 10.8s

MLPClassifier(activation='relu', alpha = 1e-05, batch_size = 'auto', beta_1 = 0.9,
              beta_2 = 0.999, early_stopping = False, epsilon = 1e-08,
              hidden_layer_sizes = [250, 250], learning_rate = 'constant', 
              learning_rate_init = 0.001, max_iter = 200, momentum = 0.9,
              n_iter_no_change = 10, nesterovs_momentum = True, power_t = 0.5,
              random_state = 1, shuffle = True, solver = 'lbfgs', tol = 0.0001,
              validation_fraction = 0.1, verbose = False, warm_start = False)

data['pos_dnn_sk'] = model.predict(data[cols_bin])
data['strat_dnn_sk'] = data['pos_dnn_sk'] * data['returns']

data[['returns', 'strat_dnn_sk']].sum().apply(np.exp)

data[['returns', 'strat_dnn_sk']].cumsum().apply(np.exp).plot(figsize = (10, 6))
plt.title('EUR/EUS and DNN-based trading strategy (scikit-learn in-sample)')

# To avoid overfitting,  implement train-test split.

train, test = train_test_split(data, test_size = 0.5, random_state = 100)

train = train.copy().sort_index()

test = test.copy().sort_index()

# Increases number of hidden layers and hidden units
model = MLPClassifier(solver = 'lbfgs', alpha = 1e-5, max_iter = 500,
                      hidden_layer_sizes = 3 * [500], random_state = 1)

model.fit(train[cols_bin], train['direction']) # %time 1 min 24 s

MLPClassifier(activation = 'relu', alpha = 1e-05, batch_size = 'auto', beta_1 = 0.9,
              beta_2 = 0.999, early_stopping = False, epsilon = 1e-08,
              hidden_layer_sizes = [500, 500, 500], learning_rate = 'constant',
              learning_rate_init = 0.001, max_iter = 500, momentum = 0.9,
              n_iter_no_change = 10, nesterovs_momentum = True, power_t = 0.5,
              random_state = 1, shuffle = True, solver = 'lbfgs', tol = 0.0001,
              validation_fraction = 0.1, verbose = False, warm_start = False)

test['pos_dnn_sk'] = model.predict(test[cols_bin])
test['strat_dnn_sk'] = test['pos_dnn_sk'] * test['returns']

test[['returns', 'strat_dnn_sk']].sum().apply(np.exp)

# Seems to underperform?..
test[['returns', 'strat_dnn_sk']].cumsum().apply(np.exp).plot(figsize = (10, 6))
plt.title('EUR/USD and DNN-based trading strategy (scikit-learn, randomized train-test split)')

#%% DNN with TensorFlow
import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

fc = [tf.contrib.layers.real_valued_column('lags', dimension = lags)]

model = tf.contrib.learn.DNNClassifier(hidden_units = 3 * [500],
                                       n_classes = len(bins) + 1,
                                       feature_columns = fc)

def input_fn():
    fc = {'lags' : tf.constant(data[cols_bin].values)}
    la = tf.constant(data['direction'].apply(lambda x: 0 if x < 0 else 1).values,
                     shape=[data['direction'].size, 1])
    return fc, la

model.fit(input_fn = input_fn, steps = 250) # %time 29.7s

model.evaluate(input_fn = input_fn, steps = 1) # Binary predictions (0,1)...

pred = np.array(list(model.predict(input_fn = input_fn))) # Binary predictions (0,1)...

data['pos_dnn_tf'] = np.where(pred > 0, 1, -1) #...need to be transformed to market positions (-1, +1)

data['strat_dnn_tf'] = data['pos_dnn_tf'] * data['returns']

data[['returns', 'strat_dnn_tf']].sum().apply(np.exp)
#returns         1.110278
#strat_dnn_tf    2.114755
#dtype: float64

data[['returns', 'strat_dnn_tf']].cumsum().apply(np.exp).plot(figsize = (10, 6))

plt.title('EUR/USD and DNN-based trading strategy (TensorFlow, in-sample)')

# Now with train-test split

model = tf.contrib.learn.DNNClassifier(hidden_units = 3 * [500],
                                       n_classes = len(bins) + 1,
                                       feature_columns = fc)

data = train # from the scikit learn example

model.fit(input_fn = input_fn, steps = 2500) # %time 2min 4s

data = test # from the scikit learn example
model.evaluate(input_fn = input_fn, steps = 1)

pred = np.array(list(model.predict(input_fn = input_fn)))

test['pos_dnn_tf'] = np.where(pred > 0, 1, -1)
test['strat_dnn_tf'] = test['pos_dnn_tf'] * test['returns']

test[['returns', 'strat_dnn_sk', 'strat_dnn_tf']].sum().apply(np.exp)

test[['returns', 'strat_dnn_sk', 'strat_dnn_tf']].cumsum().apply(np.exp).plot(figsize = (10, 6))
plt.title('EUR/USD and DNN-based trading strategy (TensorFlow, randomized train-test split)')
