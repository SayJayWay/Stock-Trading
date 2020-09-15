# Bayesian statistics is important in empirical finance.
# Most popular interpretation of Bayes' formula is the "diachronic interpretation"
# This mainly states that over time one learns new information about certain
# variables or parameters of interest (i.e. mean return of a time series)

# p(H|D) = (p(H) * p(D|H))/p(D)
#           where   p(H) - "prior" probability
#                   p(D) - probability for data under any hypothesis, called the
#                          "normalizing constant"
#                   p(D|H) - probability of data under hypothesis "H"
#                   p(H|D) - posterior probability (i.e. after one has seen the data)

# PyMC3 package can be used to technically implement Bayesian statistics and 
# probabilistic programming.  Consider the following example based on noisy data
# around a straight line.  First, a linear orindary least-squares regression
# is implemented

#%% Bayesian regression
import numpy as np
import pandas as pd
import datetime as dt
from pylab import mpl, plt

plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'
np.random.seed(1000)

x = np.linspace(0, 10, 50)
y = 4 + 2 * x + np.random.standard_normal(len(x)) * 2

reg = np.polyfit(x, y, 1)

plt.figure(figsize = (10, 6))
plt.scatter(x, y, c = y, marker = 'v', cmap = 'coolwarm')
plt.plot(x, reg[1] + reg[0] * x, lw = 2.0)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')

#%% A major element of Bayesian regression is "Markov chain Monte Carlo (MCMC)
# sampling, which is essentially drawing multiple samples in a systematic, 
# automated way.

# There are 3 different functions that will need to be called
# find_MAP() - finds starting point for sampling algorithm by deriving the local
#               maximum a posteriori point
# NUTS()     - implements "efficient No-U-Turn Sampler with dual averaging" (NUTS)
#               algorithm for MCMC sampling given assumed priors
# sample()   - draws number of samples given starting value from find_MAP() and
#               the optimal step size from NUTS algorithm

import pymc3 as pm

# Note, this took 104 seconds to do
with pm.Model() as model:
    # model
    alpha = pm.Normal('alpha', mu = 0, sd = 20) # Defining the priors
    beta = pm.Normal('beta', mu = 0, sd = 10) # Defining the priors
    sigma = pm.Uniform('sigma', lower = 0, upper = 10) # Defining the priors
    y_est = alpha + beta * x # Specifies the linear regression
    likelihood = pm.Normal('y', mu = y_est, sd = sigma, observed = y) # Defines likelihood
    
    # inference
    start = pm.find_MAP() # Finds the starting value by optimization
    step = pm.NUTS() # Instantiates the MCMC algorithm
    trace = pm.sample(100, tune = 1000, cores = 1, start = start, progressbar = True) # Draws posteriod samples using NUTS
  
#%%
pm.summary(trace) # Shows summary statistics from sampling

# Output
#        mean     sd  hdi_3%  hdi_97%  ...  ess_sd  ess_bulk  ess_tail  r_hat
#alpha  4.058  0.476   3.129    4.911  ...    88.0      94.0     120.0   1.02
#beta   1.991  0.084   1.827    2.129  ...    96.0      97.0     151.0   1.01
#sigma  1.762  0.186   1.429    2.096  ...   119.0     119.0     115.0   1.01

trace[0] # Estimates from the first sample
# {'alpha': 4.906065607596319,
#  'beta': 1.8295274683208118,
#  'sigma_interval__': -1.7898448413486234,
#  'sigma': 1.4309174743720585}

# trace[0] only shows the first sample...to get a better visualization of the
# procedure of sampling, we can create a plot

pm.traceplot(trace, lines = {'alpha' : 4, 'beta' : 2, 'sigma' : 2})

plt.figure(figsize = (10, 6))
plt.scatter(x, y, c = y, marker = 'v', cmap = 'coolwarm')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Regression lines based on different estimates')

# Plots single regression lines
for i in range(len(trace)):
    plt.plot(x, trace['alpha'][i] + trace['beta'][i] * x)
    
#%% Applying Bayesian regression with real-world data financial time series data
# for two ETFs GLD and GDX
    
GLD_raw = pd.read_csv('../stock_dfs/GLD.csv', index_col = 0, parse_dates = True)
GDX_raw = pd.read_csv('../stock_dfs/GDX.csv', index_col = 0, parse_dates = True)

data = pd.DataFrame({'GLD' : GLD_raw['Adj Close'],
                     'GDX' : GDX_raw['Adj Close']}, index = GLD_raw.index)
    
data = data / data.iloc[0] # Normalizing data to a starting value of 1
data.iloc[-1] / data.iloc[0] - 1 # Calculates relative performance
data.corr() # Correlation between two instruments

# Line plot of normalized data values
plt.figure(figsize = (10, 6))
plt.plot(data)

# Plotting values of GDX VS GLD

# Converts DatetimeIndex object to matplotlib dates
mpl_dates = mpl.dates.date2num(data.index.to_pydatetime())
plt.figure(figsize = (10, 6))
plt.scatter(data['GDX'], data['GLD'], c = mpl_dates, marker = 'o', cmap = 'coolwarm')
plt.xlabel('GDX')
plt.ylabel('GLD')
plt.colorbar(ticks = mpl.dates.DayLocator(interval = 250),
             format = mpl.dates.DateFormatter('%d %b %y'))
plt.title('Scatter plot of GLD VS GDX prices')

#%% Bayesian regression on the basis of these two time series

with pm.Model() as model:
    alpha = pm.Normal('alpha', mu = 0, sd = 20)
    beta = pm.Normal('beta', mu = 0, sd = 20)
    sigma = pm.Uniform('sigma', lower = 0, upper = 50)
    
    y_est = alpha + beta * data['GDX'].values
    likelihood = pm.Normal('GLD', mu=y_est, sd = sigma, observed=data['GLD'].values)
    
    start = pm.find_MAP()
    step = pm.NUTS()
    trace = pm.sample(250, tune = 2000, start = start, progressbar=True, cores = 1)

pm.summary(trace)

fig = pm.traceplot(trace)

# Adding the regression lines to the scatter plot, however, all the lines are close
# to each other
plt.figure(figsize = (10, 6))
plt.scatter(data['GDX'], data['GLD'], c = mpl_dates, marker = 'o', cmap = 'coolwarm')
plt.xlabel('GDX')
plt.ylabel('GLD')
for i in range(len(trace)):
    plt.plot(data['GDX'],
             trace['alpha'][i] + trace['beta'][i] * data['GDX'])
plt.colorbar(ticks = mpl.dates.DayLocator(interval = 250),
             format = mpl.dates.DateFormatter('%d %b %y'))
plt.title('Multiple Bayesian regression lines through GDX and GLD data')

# This figure reveals a major drawback of the regression approach used: it does
# not take into account evolutions over time.  That is, the most recent data is
# treated the same way as the oldest data.

#%% It is more useful when Bayesian approach is seen as diachronic (i.e. new data
# revealed over time allows for better regression and estimates through updating
# or learning).  To incorporate this, assume regression parameters are not only random
# and distributed, but they follow some "random walk" over time (same theory used
# when changing from random variables to stochastic processes (which are essentially
# ordered sequences of random variables))

from pymc3.distributions.timeseries import GaussianRandomWalk

subsample_alpha = 50
subsample_beta = 50

model_randomwalk = pm.Model()
with model_randomwalk:
    # Defines priors for random walk parameters
    sigma_alpha = pm.Exponential('sig_alpha', 1. / .02, testval=.1) 
    sigma_beta = pm.Exponential('sig_beta', 1. / .02, testval=.1)
    
    # Models for random walk
    alpha = GaussianRandomWalk('alpha', sigma_alpha ** -2,
                               shape = int(len(data) / subsample_alpha))
    beta = GaussianRandomWalk('beta', sigma_beta ** -2,
                              shape = int(len(data) / subsample_beta))
    
    # Brings parameter vectors to interval length
    alpha_r = np.repeat(alpha, subsample_alpha)
    beta_r = np.repeat(beta, subsample_beta)
    
    # Defines regression model
    regression = alpha_r + beta_r * data['GDX'].values[:1150]
    
    # Prior for the standard deviation
    sd = pm.Uniform('sd', 0, 20)
    
    # Defines likelihood with mu from regression results
    likelihood = pm.Normal('GLD', mu = regression, sd = sd, observed = data['GLD'].values[:1150])
    
import scipy.optimize as sco

with model_randomwalk:
    start = pm.find_MAP(vars = [alpha, beta], fmin = sco.fmin_l_bfgs_b)
    step = pm.NUTS(scaling = start)
    trace_rw = pm.sample(250, tune = 1000, start = start, progressbar = True, cores = 1)
    
pm.summary(trace_rw).head() # Summary stats per interval (first five and alpha only)

# plotting
sh = np.shape(trace_rw['alpha'])

#Creates list of dates to match  number of intervals
part_dates = np.linspace(min(mpl_dates), max(mpl_dates), sh[1])
index = [dt.datetime.fromordinal(int(date)) for date in part_dates]

# Collects relevant parameter time series in two dataframe objects
alpha = {'alpha_%i' % i: v for i, v in enumerate(trace_rw['alpha']) if i < 20}
beta = {'beta_%i' % i: v for i, v in enumerate(trace_rw['beta']) if i < 20}
df_alpha = pd.DataFrame(alpha, index=index)
df_beta = pd.DataFrame(beta, index=index)

ax = df_alpha.plot(color = 'b', style = '-.', legend = False, lw = 0.7, figsize = (10, 6))

df_beta.plot(color = 'r', style = '-.', legend = False, lw = 0.7, ax=ax)

plt.ylabel('alpha/beta')
plt.title('Selected parameter estimates over time')

#** Note that the analyses done here are based on normalized price data as it is
# easier to interpret the graphs.  In real world applications, one would instead
# rely on return data, for instance, to ensure stationarity of the time series data

# Can now see how the regression is updates over time and how it improves drastically
# (vs the previous graph/data)

plt.figure(figsize = (10, 6))
plt.scatter(data['GDX'], data['GLD'], c = mpl_dates, marker = 'o', cmap = 'coolwarm')
plt.colorbar(ticks = mpl.dates.DayLocator(interval = 250),
             format = mpl.dates.DateFormatter('%d %b %y'))
plt.xlabel('GDX')
plt.ylabel('GLD')
x = np.linspace(min(data['GDX']), max(data['GDX']))

# Plotting the regression lines for all the time intervals of length 50
for i in range(sh[1]):
    alpha_rw = np.mean(trace_rw['alpha'].T[i])
    beta_rw = np.mean(trace_rw['beta'].T[i])
    plt.plot(x, alpha_rw + beta_rw * x, '--', lw = 0.7, color = plt.cm.coolwarm(i / sh[1]))