import pandas as pd
import math
import numpy as np
from pylab import mpl, plt
import scipy.stats as scs
import statsmodels.api as sm

#%%
def print_statistics(array):
    ''' Prints selected statistics.
    
    Parameters
    ==========
    array: ndarray
            object to generate statistics on
    '''
    
    sta = scs.describe(array) # Gives back important stats for the function
    print('%14s %15s' % ('statistics', 'value'))
    print(30 * "-")
    print('%14s %15.5f' % ('size', sta[0]))
    print('%14s %15.5f' % ('min', sta[1][0]))
    print('%14s %15.5f' % ('max', sta[1][1]))
    print('%14s %15.5f' % ('mean', sta[2]))
    print('%14s %15.5f' % ('std', np.sqrt(sta[3])))
    
    # Skew is measure of symmetry or, more precisely, lack of asymettry
    print('%14s %15.5f' % ('skew', sta[4]))
    
    # Kurtosis is a measure of whether the data are heavy-tailed or light-tailed 
    # relative to a normal distribution. data sets with high kurtosis tend to have
    # heavy tails, or outliers.
    print('%14s %15.5f' % ('kurtosis', sta[5]))
 
#%% 
AAPL_raw = pd.read_csv('../stock_dfs/AAPL.csv', index_col = 0, parse_dates = True).dropna()
MSFT_raw = pd.read_csv('../stock_dfs/MSFT.csv', index_col = 0, parse_dates = True).dropna()
SPY_raw = pd.read_csv('../stock_dfs/SPY.csv', index_col = 0, parse_dates = True).dropna()
GLD_raw = pd.read_csv('../stock_dfs/GLD.csv', index_col = 0, parse_dates = True).dropna()
TSLA_raw = pd.read_csv('../stock_dfs/MSFT.csv', index_col = 0, parse_dates = True).dropna()

symbols = ['AAPL', 'MSFT', 'SPY', 'GLD', 'TSLA']

data = pd.DataFrame({'AAPL' : AAPL_raw['Adj Close'],
                     'MSFT' : MSFT_raw['Adj Close'],
                     'SPY' : SPY_raw['Adj Close'],
                     'GLD' : GLD_raw['Adj Close'],
                     'TSLA' : TSLA_raw['Adj Close']})

data = data.dropna()

(data / data.iloc[0] * 100).plot(figsize=(10, 6))
plt.title('Normalized prices of financial instruments over time')
plt.ylabel('Normalized price')

log_returns = np.log(data / data.shift(1))

log_returns.hist(bins = 50, figsize=(10, 8))

for symbol in symbols:
    print('\nResults for symbol {}'.format(symbol))
    print(30 * '-')
    log_data = np.array(log_returns[symbol].dropna())
    print_statistics(log_data)

# Can see that kurtosis is far from normal for all data sets
sm.qqplot(log_returns['SPY'].dropna(), line = 's')
plt.title('QQ-plot for SPY log returns')
plt.xlabel('theoretical quantiles')
plt.ylabel('sample quantiles')

sm.qqplot(log_returns['MSFT'].dropna(), line = 's')
plt.title('QQ-plot for MSFT log returns')
plt.xlabel('theoretical quantiles')
plt.ylabel('sample quantiles')

# Can see that QQ plots show fat left and right tails.  This term refers to a (frequency)
# distribution where large -ve and +ve are observed more often than a normal
# distribution, showing evidence for a fat-tailed distribution.

#%% Portfolio Optimization
# Mean-variance portfolio theory (MPT) is a major cornerstone of financial theory.
# "By looking only at mean and variance, we are assuming other statistics are
#   necessary to describe the distrubution at end-of-period wealth.  Unless investors
#   have a special type of utility function (quadratic utility function), one must
#   assume that returns have a normal distribution, which can be completely described
#   by mean and variance.

# Basic idea of MPT is to make use of "diversification" to achieve a minimal portfolio
# risk given a target return level or a maximum portfolio return given a certain
# level of risk.  One would expect such diversification effects for the right 
# combination of a larger number of assets and a certain diversity in the assets

# To convey the basics ideas, we will look at 4 fiancial instruments

symbols = ['AAPL', 'MSFT', 'SPY', 'GLD'] # Our portfolio

data = pd.DataFrame({'AAPL' : AAPL_raw['Adj Close'],
                     'MSFT' : MSFT_raw['Adj Close'],
                     'SPY' : SPY_raw['Adj Close'],
                     'GLD' : GLD_raw['Adj Close'],})
    
rets = np.log(data / data.shift(1))

rets.hist(bins = 40, figsize = (10, 8))
plt.suptitle('Histograms of log returns of financial instruments')

rets.mean() * 252 # Annualized mean returns 
rets.cov() * 252 # Annualized covariance matrix

#%% Basic Theory
# It is assumed that an investor is not allowed to set up short positions, which
# implies that 100% of the investor's wealth has to be divided among the available
# instruments in such a way that all positions are long AND that the positions
# add up to 100%.  Given the four instruments, one could, for example invest 25%
# of the available wealth into each.  Following code generates four uniformly
# distributed random numbers between 0 and 1 and then normalizes the values such
# that the sum of all values equals 1

noa = len(symbols)
weights = np.random.random(noa) # Random portfolio weights
weights /= np.sum(weights) # Normalized to 1 or 100%

weights.sum() # Should be 1.0

###### Next lines of code are main tools for MPT #######
# Annualized portfolio return given porfolio weights
np.sum(rets.mean() * weights) * 252

# Annualized portfolio variance given portfolio weights
np.dot(weights.T, np.dot(rets.cov() * 252, weights))

# Annualized portfolio volatility given portfolio weights
math.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))

#%% Implementing Monte Carlo simulations to generate random portfolio weight
# vectors on a larger scale.  For every simulated allocation, the code records
# the resulting expected portfolio return and variance.  Of paramount interest
# to investors is what risk-return profiles are possible for a given set of
# financial instruments and their statistical characteristics.

def port_ret(weights):
    return np.sum(rets.mean() * weights) * 252

def port_vol(weights):
    return np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))

prets = []
pvols = []
for p in range(2500): # Monte Carlo simulation of portfolio weights
    weights = np.random.random(noa) # Monte Carlo simulation of portfolio weights
    weights /= np.sum(weights) # Monte Carlo simulation of portfolio weights
    prets.append(port_ret(weights)) # Resulting statistics
    pvols.append(port_vol(weights)) # Resulting statistics
prets = np.array(prets)
pvols = np.array(pvols)

plt.figure(figsize = (10, 6))
plt.scatter(pvols, prets, c = prets / pvols, marker = 'o', cmap = 'coolwarm')
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label = 'Sharpe ratio')
plt.title('Expected return and volatility for random portfolio weights')

# From graph, it is clear that not all weight distributions perform well
# when measured in terms of mean and volatility.  For exmaple, at 15%, there
# are multiple estimated returns.  As an investor, one is generally interested
# in the maximum return given a fixed risk level or minimum risk given a fixed
# return expectation.  This set of portfolios then makes up the "efficient frontier"
# The graph also shows the Sharpe ratio (return of investment compared to its risk).
# The greater the value of the Sharpe ratio, the more attractive the risk-adjusted
# return.

#%% Optimal Portfolios
# We want maximization of the Sharpe Ratio, or more specifically, we want to
# minimize the negative value of the Sharpe ratio.  The constraint is that all
# parameters (weights) add up to 1.  The values will be provided to the 
# minimization function as a tuple of tuples.

# The only input that is missing for a call of the optimization function is a
# starting parameter list (initial guess for the weights vector).  For now,
# we will use an equal distribution of weights.

import scipy.optimize as sco

def min_func_sharpe(weights):  # Function to be minimized
    return -port_ret(weights) / port_vol(weights)

cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # Equality constraight

bnds = tuple((0, 1) for x in range(noa)) # Bounds for the parameters

eweights = np.array(noa * [1. / noa,]) # Equal weights vector

min_func_sharpe(eweights) # Function to be minimized

opts = sco.minimize(min_func_sharpe, eweights, method = 'SLSQP', bounds = bnds,
constraints = cons) # %time 79.8ms -> Optimization/minimization of min_func_sharpe())

# One can obtain the results object by providing the key of interest (x)
opts['x'].round(3) # Optimal portfolio weights
returns = port_ret(opts['x']).round(3) # 0.198 -> Resulting portfolio return
volatility = port_vol(opts['x']).round(3) # 0.133 -> Resulting portfolio volatility

max_Sharpe = returns/volatility # 1.4887218045112782

#%% Minimization of Variance of Portfolio (this is the same as minimizing the
# volatility)

# Minimizing volatility function
optv = sco.minimize(port_vol, eweights, method = 'SLSQP', bounds = bnds,
                    constraints = cons)

# Optimal Parameters
optv['x'].round(3)

# Annualized volatility at the parameters
volatility = port_vol(optv['x']).round(3)

# Annualized returns at the parameters
returns = port_ret(optv['x']).round(3)

# Sharpe Ratio at the parameters
Sharpe = returns/volatility

# This is the sharpe ratio for the 'minimum volatility' or 'minimum variance portfolio'

#%% Efficient Frontier
# This is a line representing the maximum return for every given risk level.
# To get this, one must iterate over multiple starting conditions.  The approach
# taken is to fix a target return level and to derive for each such level the
# portfolio weights that lead to the minimum volatility value.  For the
# optimization, this leads to two conditions: one for the target return level,
# 'tret' and one for the sum of the portfolio weights as before.  Boundary
# values for each parameter stay the same.  When iterating over different target
# return levels (trets), one condition for the minimization changes.  This is why
# the constraints dictionary updates during every loop

# Two binding constraints for the efficient frontier
cons = ({'type': 'eq', 'fun': lambda x: port_ret(x) - tret},
         {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

bnds = tuple((0, 1) for x in weights)

#%%time
trets = np.linspace(0.125, 0.32, 50)
tvols = []
for tret in trets:
    # Minimization of portfolio volatility for different target returns
    res = sco.minimize(port_vol, eweights, method = 'SLSQP', bounds = bnds,
                       constraints = cons)
    tvols.append(res['fun'])
tvols = np.array(tvols)

plt.figure(figsize = (10, 6))
plt.scatter(pvols, prets, c = prets / pvols, marker = '.', alpha = 0.8,
            cmap = 'coolwarm')
plt.plot(tvols, trets, 'b', lw = 4.0)
plt.plot(port_vol(opts['x']), port_ret(opts['x']), 'y*', markersize = 15.0)
plt.plot(port_vol(optv['x']), port_ret(optv['x']), 'r*', markersize = 15.0)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label = 'Sharpe Ratio')

#%% Capital Markets line
# A riskless investment opportunity is "cash" or "cash accounts".  Money held in 
# a cash account with a large bank can be considered "riskless".  The downside 
# is the little to no yield.  Taking into account such a riskless asset enhances
# the efficient investment opportunity set for investors.  The idea is that
# investors first determine an efficient portfolio of risky assets and then add
# the riskless asset to the mix.  By adjusting the proportions, it is possible
# to achieve any risk-return profile that lies on a straight line between the
# riskless asset and the efficient portfolio

import scipy.interpolate as sci

ind = np.argmin(tvols) # Index position of minimum volatility porfolio
evols = tvols[ind:] # Relevant portfolio volatility
erets = trets[ind:] # Relevant portfolio return values

tck = sci.splrep(evols, erets) # Cubic splines interpolation on these values

def f(x):
    ''' Efficient frontier function (splines approximation). '''
    return sci.splev(x, tck, der = 0)

def df(x):
    ''' First derivative of efficient frontier function. '''
    return sci.splev(x, tck, der = 1)

def equations(p, rf = 0.01):
    # Equations describing capital market line (CML) pg 427
    eq1 = rf - p[0] 
    eq2 = rf + p[1] * p[2] - f(p[2])
    eq3 = p[1] - df(p[2])
    return eq1, eq2, eq3

opt = sco.fsolve(equations, [0.01, 0.5, 0.15]) # Solving equations for given initial values

np.round(equations(opt, 6))

plt.figure(figsize = (10, 6))
plt.scatter(pvols, prets, c = (prets - 0.01) / pvols, marker = '.', cmap = 'coolwarm')
plt.plot(evols, erets, 'b', lw = 4.0)
cx = np.linspace(0.0, 0.3)
plt.plot(cx, opt[0] + opt[1] * cx, 'r', lw = 1.5)
plt.plot(opt[2], f(opt[2]), 'y*', markersize = 15.0)
plt.grid(True)
plt.axhline(0, color = 'k', ls = '--', lw = 2.0)
plt.axvline(0, color = 'k', ls = '--', lw = 2.0)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label = 'Sharpe ratio')
plt.title('Capital market line and tangent portfolio (star) for risk-free rate of 1%')

#%% Portfolio weights of the optimal (tangent) portfolio are as follows.  Only
# three of the four assets are in the mix

# Binding constraints for tangent portfolio (gold star in previous figure)
cons = ({'type': 'eq', 'fun': lambda x: port_ret(x) - f(opt[2])},
         {'type' : 'eq', 'fun': lambda x: np.sum(x) - 1})

res = sco.minimize(port_vol, eweights, method = 'SLSQP', bounds = bnds,
                   constraints = cons)

res['x'].round(3) # Portfolio weights for this particular portolio
returns = port_ret(res['x'])
volatility = port_vol(res['x'])
sharpe = returns/volatility

