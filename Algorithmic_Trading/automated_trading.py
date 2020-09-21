# =============================================================================
# Kelly criterion in Binomial Setting
# =============================================================================
# Most financial institutions will agree that the "maximization of long-term
# wealth" is a good candidate objective.  This is what was kept in mind when the
# Kelly creterion was created.

# Kelly criterion = f* = p - q
#           where f* == fraction of capital one should bet
#                 p  == percent change of winning
#                 q  == percent change of losing
# Example: p = 0.55
#   Then f* = 0.55 - 0.45 = 0.1 -> Optimal fraction is 10%

import math
import time
import numpy as np
import pandas as pd
import datetime as dt
import cufflinks as cf
from pylab import plt

np.random.seed(1000)
plt.style.use('seaborn')

# Simulat 50 series ith 100 coin tosses per series.
p = 0.55 # Probability for heads
f = p - (1 - p) # Optimal Fraction as per Kelly criterion
I = 50 # Number of series to be simulated
n = 100 # Number of trials per series

def run_simulation(f):
    c = np.zeros((n, I))
    c[0] = 100 # Starting capital = 100
    for i in range(I): # series of simulations
        for t in range(1, n): # trials in each series
            o = np.random.binomial(1, p) # Simulates coin toss 1 time, with p probability
            if o > 0: # i.e. if heads
                c[t, i] = (1 + f) * c[t - 1, i] # Add to capital
            else: # if tails
                c[t, i] = (1 - f) * c[t - 1, i] # Subtract from capital
    return c

c_1 = run_simulation(f)

plt.figure(figsize = (10, 6))
plt.plot(c_1, 'b', lw = 0.5)
plt.plot(c_1.mean(axis = 1), 'r', lw = 2.5)
plt.title('50 Simulated series with 100 trials each (red line = average)')

# Repeating simulation for different values of f.  Can see that a lower fraction
# leads to a higher average capital at te end of the simulation (f = 0.25), or to
# a much lower average capital (f = 0.5).  Where the fraction f is higher, the 
# volatility increases considerably

c_2 = run_simulation(0.05)
c_3 = run_simulation(0.25)
c_4 = run_simulation(0.5)

plt.figure(figsize = (10, 6))
plt.plot(c_1.mean(axis = 1), 'r', label='$f^*=0.1$')
plt.plot(c_2.mean(axis = 1), 'b', label='$f^*=0.05$')
plt.plot(c_3.mean(axis = 1), 'y', label='$f^*=0.25$')
plt.plot(c_4.mean(axis = 1), 'm', label='$f^*=0.5$')
plt.legend(loc = 0)
plt.title('Average capital over time for different functions')

#%% Assume stock market setting in which the relevant stock can take on only two
# values after a period of one year from today, given its known value today.

# The optimal fraction can be given by F* = (mu-r)/(sigma^2)

symbol = 'SPY'
raw = pd.read_csv('../stock_dfs/{}.csv'.format(symbol, index_col = 0))

data = pd.DataFrame(raw['Adj Close'])
data['returns'] = np.log(data / data.shift(1))

data.dropna(inplace = True)

mu = data.returns.mean() * 252 # Annualized return

sigma = data.returns.std() * 252 ** 0.5 # Annualized volatility

r = 0.0 # Sets risk-free rate to 0 (for simplicity)

f = (mu - r) / sigma ** 2 # Optimal Kelly fraction to be invested
# Output = 3.485132299271084 -> ~3.5 which implies using LEVERAGE..

# For simple calculations, initial capital = 1

equs = []

def kelly_strategy(f):
    global equs
    equ = 'equity_{:.2f}'.format(f)
    equs.append(equ)
    cap = 'capital_{:.2f}'.format(f)
    data[equ] = 1 # Column equity = 1
    data[cap] = data[equ] * f # Column capital = 1 * f

    for i, t in enumerate(data.index[1:]):
        t_1 = data.index[i] # Picks right DatetimeIndex value for previous values
        # Calculates new capital position given return
        data.loc[t, cap] = data[cap].loc[t_1] * math.exp(data['returns'].loc[t])
        # Adjusts equity value according to capital position performance
        data.loc[t, equ] = data[cap].loc[t] - data[cap].loc[t_1] + data[equ].loc[t_1]
        # Adjusts capital position given new equity position and fixed leverage ratio
        data.loc[t, cap] = data[equ].loc[t] * f
        
kelly_strategy(0.5 * f)
kelly_strategy(0.66 * f)
kelly_strategy(f)

print(data[equs].tail())

ax = data['returns'].cumsum().apply(np.exp).plot(figsize = (10, 6), legend = True)
data[equs].plot(ax = ax, legend = True)\

# From graph, can see that applying the "optimal" Kelly leverage leads to high
# volatility.  One would expect this.  Practitioners often reduce the leverage to,
# for example, "half Kelly", which reduces the risk (lower f = lower risk in general)

#%%
# =============================================================================
# ML-based trading strategy
# =============================================================================
# Using FXCM REST API, we can see the big/ask spread as proportional transaction
# costs.  Let's use 5-minute bars.

import fxcmpy
from pylab import plt

def get_data():
    api = fxcmpy.fxcmpy(config_file='../fxcm.cfg')
    
    data = api.get_candles('EUR/USD', period = 'm5', 
                           start = '2018-06-01 00:00:00',
                           end = '2018-06-30 00:00:00')
    return api, data
api, data = get_data()
# Average bid-ask spread
spread = (data['askclose'] - data['bidclose']).mean()

# Mid close prices from ask and bid close prices
data['midclose'] = (data['askclose'] + data['bidclose']) / 2

# Average proportional transaction costs given the average spread and average
# mid close price
ptc = spread / data['midclose'].mean()

data['midclose'].plot(figsize = (10, 6), legend = True)
plt.title('EUR/USD exchange rate (five-minute bars)')

# Our ML-based strategy is based on lagged return data that is binarized.  i.e.
# ML algo learns from historical patterns of upward and downward movements
# whether up or down is more likely.  Code will create features data w/ values
# of 0 and 1 as well as labels data with values of +1 and -1 indicating the
# observed market direction in all cases

data['returns'] = np.log(data['midclose'] / data['midclose'].shift(1))
data.dropna(inplace = True)
lags = 5

cols = []
for lag in range(1, lags + 1):
    col = 'lag_{}'.format(lag)
    data[col] = data['returns'].shift(lag) # Creates lag return data given number of lags
    cols.append(col)
    
data.dropna(inplace = True)
# Transforms feature values to binary daya
data[cols] = np.where(data[cols] > 0, 1, 0)

# Transforms return data into directional label data
data['direction'] = np.where(data['returns'] > 0, 1, -1)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

model = SVC(C = 1, kernel = 'linear', gamma = 'auto')

split = int(len(data) * 0.80)

train = data.iloc[:split].copy()

model.fit(train[cols], train['direction'])

# In-sample accuracy from trained model (training data)
accuracy_score(train['direction'], model.predict(train[cols])) # 0.5206747582801893

test = data.iloc[split:].copy()
test['position'] = model.predict(test[cols])
# Out-of-sample accuracy from trained model (test data)
accuracy_score(test['direction'], test['position']) # 0.5419407894736842

# Hit ratio > 50% on both train and test data

#%%
# Taking into account the proportional transaction costs based on the average 
# bid-ask spread.

# Log returns for ML-based algo trading strategy
test['strategy'] = test['position'] * test['returns']

sum(test['position'].diff() != 0) # Number of trades 

test['strategy_tc'] = np.where(test['position'].diff() != 0,
                               test['strategy'] - ptc, # Subtracts ptc from return on that day
                                                       # whenever a trade takes place
                               test['strategy'])

test[['returns', 'strategy', 'strategy_tc']].sum().apply(np.exp)
#returns        0.999324
#strategy       1.026141
#strategy_tc    0.999613

test[['returns', 'strategy', 'strategy_tc']].cumsum().apply(np.exp).plot(figsize = (10, 6))
plt.title('EUR/USD exchange rate and algorithmic trading strategy')

#%% Optimized Leverage
# Since we have trading strategy's log returns data, the mean and variance 
# values can be calculated in order to derive the optimal leverage according
# to the Kelly criterion.  We can scale the numbers to annualized values:

mean = test[['returns', 'strategy_tc']].mean() * len(data) * 12 # Annualized mean returns
#returns       -0.040535
#strategy_tc   -0.023228

var = test[['returns', 'strategy_tc']].var() * len(data) * 12 # Annualized variances
#returns        0.007861
#strategy_tc    0.007855

vol = var ** 0.5 # Annualized volatilies
#returns        0.088663
#strategy_tc    0.088630

mean / var # Optimal leverage according to Kelly criterion ("full Kelly")
#returns       -5.156448
#strategy_tc   -2.956984

mean / var * 0.5 # "half Kelly"
#returns       -2.578224
#strategy_tc   -1.478492

# Using the "half Kelly" criterion, the optimal leverage for the trading 
# strategy is about 40.  This leverage is possible in FX markets.

to_plot = ['returns', 'strategy_tc']

for lev in [10, 20, 30, 40, 50]:
    label = 'lstrategy_tc_%d' % lev
    test[label] = test['strategy_tc'] * lev # Scales strategy returns for different leverage values
    to_plot.append(label)
    
test[to_plot].cumsum().apply(np.exp).plot(figsize = (10, 6))
plt.title('Performance of algo trading strategy for different leverage values')

#%% Risk Analysis
# Since leverage increases risk, it is necessary to have a more in-depth risk
# analysis.  Maximum drawdown (largest loss after recent high) and longest 
# drawdown period (period that trading strategy needs to get back to recent high)
# are calculated.  The analysis assumes leverage of 30, with the initial equity 
# position is 3,333 EUR, leading to initial position of 100,000EUR.  It also assumes
# no adjustments with regard to equity over time, no matter what the performance
# is.

equity = 3333
risk = pd.DataFrame(test['lstrategy_tc_30']) # Relevant log returns time series...

# ...scaled by initial equity
risk['equity'] = risk['lstrategy_tc_30'].cumsum().apply(np.exp) * equity

risk['cummax'] = risk['equity'].cummax() # cumulative maximum values over time

risk['drawdown'] = risk['cummax'] - risk['equity'] # Drawdown values over time

risk['drawdown'].max() # Maximum drawdown value
#Out: 714.0806803769715

t_max = risk['drawdown'].idxmax() # Point in time when it happens
# Output : Timestamp('2018-06-29 02:45:00')

# Technically a (new) high is characterized by a drawdown value of 0.  The
# drawdown period is the time between two such highs

temp = risk['drawdown'][risk['drawdown'] == 0] # Identifies highs for which drawdown must be 0

# Calculates timedelta values between all highs
periods = (temp.index[1:].to_pydatetime() - temp.index[:-1].to_pydatetime())

# Longest drawdown period in seconds
t_per = periods.max()

# Longest drawdown period in hours
t_per.seconds / 60 / 60

risk[['equity', 'cummax']].plot(figsize = (10, 6))
plt.axvline(t_max, c = 'r', alpha = 0.5)
plt.title('Max drawdown (vertical line) and drawdown periods (horizontal lines)')

#%% Calculating VaR.  It is quoted as a currency amount and represents the max
# loss to be expected given both a certain time horizon and confidence level.
# Following is based on log returns of equity position for the leveraged trading
# strategy over time for different CIs.

import scipy.stats as scs

percs = np.array([0.01, 0.1, 1. , 2.5, 5.0, 10.0]) # % percentages to be used

risk['returns'] = np.log(risk['equity'] / risk['equity'].shift(1))

# VaR based on percentage values
VaR = scs.scoreatpercentile(equity * risk['returns'], percs)

# Translates percent values to CIs and Var Values to positive values for printing
def print_var():
    print('%16s %16s' % ('Confidence level', 'Value-at-Risk'))
    print(33 * '-')
    for pair in zip(percs, VaR):
        print('%16.2f %16.3f' % (100 - pair[0], -pair[1]))
        
print_var()

# Re-evaluating VaR values for a time horizon of 1-hour by resampling the original
# DataFrame object.  In effect, VaR values are increased for all CIs but the
# highest one.

# Resamples data from 5-minute bars to 1-hr bars
hourly = risk.resample('1H', label = 'right').last()
hourly['returns'] = np.log(hourly['equity'] / hourly['equity'].shift(1))

# Recalculate VaR values for resampled data
VaR = scs.scoreatpercentile(equity * hourly['returns'], percs)

print_var()

#%% Persisting the Model Object
# Once algo trading strategy is 'accepted' based on the backtesting, leveraging, 
# and risk analysis results, the model object might be persisted for later use
# in deployment.  It emobdies now "the" ML-based trading strategy or "the" trading
# algorithm

import pickle

pickle.dump(model, open('algorithm.pkl', 'wb')) # Saving model as a pickle

# So far, the trading algorithm tested is an "offline algorithm"  such algos
# use a complete data set to solve a problem at hand.  In practice, when deploying
# the algorithm in financial markets, it must consume data piece-by-piece as it
# arrives to predict the direction of the market movement for the next time
# interval (bar).  Thi ssection makes use of the predicted model object determined
# previously and embeds it into a streaming data environment.

# Following will be addressed:
#       Tick data - arrives in real time and is to be processed in real time
#       Resampling - Tick data to be resampled to appropriate bar size  given
#                   trading algo
#       Prediction - Algo generates prediction for direction of the market 
#                   movement over the relevant time interval that by nature
#                   lies in the future
#       Orders - Order is placed or position is kept based on current position
#                   and prediction ("signal")

algorithm = pickle.load(open('algorithm.pkl', 'rb'))

# DataFrame columns to be shown
sel = ['tradeId', 'amountK', 'currency', 'grossPL', 'isBuy']

def print_positions(pos):
    print('\n\n' + 50 * '=')
    print('Going {}.\n'.format(pos))
    time.sleep(1.5) # Waits for order to be executed and reflected in open positions
    print(api.get_open_positions()[sel]) # Print open positions
    print(50 * '=' + '\n\n')
    
# Setting the parameters
symbol = 'EUR/USD'
bar = '15s'
amount = 100 # Amount, in thousands, to be traded
position = 0 # Initial position ('neutral')
min_bars = lags + 1 # Minimum number of resampled bars required for first prediction
                    # and trade to be possible
df = pd.DataFrame() # For later resampled data

# Transforms trading algorithm into real-time context
def automated_strategy(data, dataframe):
    global min_bars, position, df
    ldf = len(dataframe)
    df = dataframe.resample(bar, label = 'right').last().ffill()
    if ldf % 20 == 0:
        print('%3d' % len(dataframe), end = ',')
        
    if len(df) > min_bars:
        min_bars = len(df)
        df['Mid'] = df[['Bid', 'Ask']].mean(axis = 1)
        df['Returns'] = np.log(df['Mid'] / df['Mid'].shift(1))
        df['Direction'] = np.where(df['Returns'] > 0, 1, -1)
        # Picks relevant feature value for all lags..
        features = df['Direction'].iloc[-(lags + 1):-1] 
        #..reshapes to a form that model can use for prediction
        features = features.values.reshape(1, -1) # 
        signal = algorithm.predict(features)[0] # Generates prediction value (-1 or +1)
        
        # Conditions to enter (keep) long position
        if position in [0, -1] and signal == 1:
            api.create_market_buy_order(symbol, amount - position * amount)
            position = 1
            print_positions('LONG')
            
        # Conditions to enter (keep) short position
        elif position in [0, 1] and signal == -1:
            api.create_market_sell_order(symbol, amount + position * amount)
            position = -1
            print_positions('SHORT')
            
    # The condigion to stop trading and close out any open position (arbitrarily)
    # defined based on number of ticks retrieved
    if len(dataframe) > 350:
        api.unsubscribe_market_data('EUR/USD')
        api.close_all()

# =============================================================================
# Infrastructure and Deployment
# =============================================================================
# When deploying an automated algorithmic trading strategy with real funds, the
# infrastructure should satisfy the following conditions
#        Reliability - i.e. automatic backups, redundancy of drives and web 
#                    connections, etc
#                    
#        Performance - Depends on amount of data being processed and the
#                    computational demand the algos generate.  Must have enough
#                    CPU cores, RAM and storage (SSD), in addition, the web
#                    connections should be sufficiently fast
#        Security - The OS and applications run should be protected by strong
#                    passwords as well as SSL encryption; the hardware should be
#                    protected from fire, water, and unauthorized physical access
# In general, these can only be satisfied from professional data center or cloud
# provider.  From development/ testing POV, Droplet from DigitalOcean is enough
# to get started.
# If using live strategies, a simple loss of connection or brief power outage
# can lead to unintended open positions or data set corruption (due to missing
# out on real-time tick data))
