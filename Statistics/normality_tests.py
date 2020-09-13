# Benchmark cases

import math
import numpy as np
import scipy.stats as scs
import statsmodels.api as sm
from pylab import mpl, plt

plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

# Set up Generation of Monte Carlo paths
def gen_paths(S0, r, sigma, T, M, I):
    ''' Generate Monte Carlo paths for geometric Brownian motion.
    
    Parameters
    ==========
    S0: float
        initial stock/index value
    r: float
        constant short rate
    T: float
        final time horizon
    M: int
        number of time steps/intervals
    I: int
        number of paths to be simulated
    
    Returns
    =======
    paths: ndarray, shape (M + 1, I)
        simulated paths given the parameters
    '''
    dt = T / M
    paths = np.zeros((M + 1, I))
    paths[0] = S0
    for t in range(1, M + 1):
        rand = np.random.standard_normal(I)
        rand = (rand - rand.mean()) / rand.std() # Matching 1st and 2nd moment
        
        # Vectorized Euler discretization of geometric Brownian motion
        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt +
                                         sigma * math.sqrt(dt) * rand)
    
    return paths
        
S0 = 100. # Initial stock price
r = 0.05 # constant short rate
sigma = 0.2 # volatility factor
T = 1.0 # Time horizon in year fractions
M = 50 # number of time intervals
I = 250000 # Number of simulated processes
np.random.seed(1000)

paths = gen_paths(S0, r, sigma, T, M, I)

S0 * math.exp(r * T) # Expected value
paths[-1].mean() # Average simulated value

plt.figure(figsize = (10, 6))
plt.plot(paths[:, :10])
plt.xlabel('time')
plt.ylabel('index level')
plt.title('Ten simulated paths of geometric Brownian Motion')

log_returns = np.log(paths[1:] / paths[:-1])

log_returns[:, 0].round(4) # Can see that one will have positive and negative
# returns day by day relative to most recent wealth position
        
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
    
print_statistics(log_returns.flatten())

log_returns.mean() * M + 0.5 * sigma ** 2 # Annualized mean log return after correction for Itô term
log_returns.std() * math.sqrt(M) # Annualized volatility (annualized std of log returns)

# Comparing frequency distribution of simulated log returns VS probability density function
plt.figure(figsize = (10, 6))
plt.hist(log_returns.flatten(), bins = 70, normed = True, label = 'frequency', color= 'b')
plt.xlabel('log return')
plt.ylabel('frequency')
x = np.linspace(plt.axis()[0], plt.axis()[1])
plt.plot(x, scs.norm.pdf(x, loc = r /M, scale = sigma / np.sqrt(M)),
         'r', lw = 2.0, label = 'pdf')
plt.legend()
plt.title('Histogram of log returns of geometric brownian motion and normal density function')
# Obviously there is a good fit

#%% Can also "test" for normality using quantile-quantile (QQ) plots.  Sample
# quantile values are compared to theoretical quantile values.  For normally 
# distributed sample data sets, such a plot would have the absolute majority
# of the quantile values (dots) lying on a straight line

sm.qqplot(log_returns.flatten()[::500], line = 's')
plt.xlabel('theoretical quantiles')
plt.ylabel('sample quantiles')
plt.title('Quantile-quantile plot for log returns of geometric Brownian motion')

#%% Graphical approaches, in general, can not replace more rigorous testing procedures
# We can actually use statistical tests to test normality

# This function will test:
#           skewness (skewtest()) - sample is 'normal' if value close to 0
#           kurtosis (kurtosistest()) - sample is 'normal' if value close to 0
#           normality (normaltest()) - Combines other two test to test for normality

# The test values indicate that the log returns of geometric Brownian motion are
# indeed normally distributed (i.e. p-values of 0.05 or above)

def normality_tests(arr):
    ''' Tests for normality distribution of given data set.
    
    Parameters
    ==========
    array: ndarray
        object to generate statistics on
    '''
    print('Skew of data set %14.3f' % scs.skew(arr))
    print('Skew test p-value %14.3f' % scs.skewtest(arr)[1])
    print('Kurt of data set %14.3f' % scs.kurtosis(arr))
    print('Kurt test p-value %14.3f' % scs.kurtosistest(arr)[1])
    print('Norm test p-value %14.3f' % scs.normaltest(arr)[1])
    
normality_tests(log_returns.flatten())

#Skew of data set          0.001
#Skew test p-value          0.430
#Kurt of data set          0.001
#Kurt test p-value          0.541
#Norm test p-value          0.607

# Finally a test to check if end-of-period values are indeed log-normally distributed.
# Both log-normally distributed end-of-period values and the transformed ones 
# ("log index level") are plotted

f, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 6))
ax1.hist(paths[-1], bins = 30)
ax1.set_xlabel('index level')
ax1.set_ylabel('frequency')
ax1.set_title('regular data')
ax2.hist(np.log(paths[-1]), bins = 30)
ax2.set_xlabel('log index level')
ax2.set_title('log data')

print_statistics(paths[-1])

#    statistics           value
#------------------------------
#          size    250000.00000
#           min        42.74870
#           max       233.58435
#          mean       105.12645
#           std        21.23174
#          skew         0.61116
#      kurtosis         0.65182

normality_tests(np.log(paths[-1]))
#
#Skew of data set         -0.001
#Skew test p-value          0.851
#Kurt of data set         -0.003
#Kurt test p-value          0.744
#Norm test p-value          0.931

# Statistics show expected behaviour
# mean value close to 105
# Skew & kurtosis close to 0 with high p-values (providing strong support for
# normal distribution))

# Shows good fit with PDF of normal distribution as expected
plt.figure(figsize = (10, 6))
log_data = np.log(paths[-1])
plt.hist(log_data, bins = 70, normed = True, label = 'observed', color = 'b')
plt.xlabel('index levels')
plt.ylabel('frequency')
x = np.linspace(plt.axis()[0], plt.axis()[1])
plt.plot(x, scs.norm.pdf(x, log_data.mean(), log_data.std()), 'r', lw = 2.0, label = 'pdf')
plt.legend()
plt.title('Histogram of log index levels of geometric Brownian motion and normal density function')

# QQplot also supports hypothesis that log index levels are normally distributed
sm.qqplot(log_data, line = 's')
plt.xlabel('theoretical quantiles')
plt.ylabel('sample quantiles')
plt.title('QQ plot for log index levels of geometric Brownian motion')