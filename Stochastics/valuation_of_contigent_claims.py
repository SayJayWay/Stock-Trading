# =============================================================================
# European Options
# =============================================================================
# Payoff of European call option on an index at maturity is given by
# h(S_T) == max(S_T - K, 0)
#           where S_T = index level at maturity date T
#                 K   = strike price

# Calculating valuation of European option using risk-neutral Monte Carlo estimator
# (pg 376)

import math
import numpy as np
import numpy.random as npr
from pylab import mpl, plt

def gen_sn(M, I, anti_paths = True, mo_match = True):
    ''' Function to generate random numbers for simulation.
    
    Parameters
    ==========
    M: int
        number of time intervals for discretization
    I: int
        number of paths to be simulated
    anti_paths: boolean
        use of antithetic variates
    mo_math: boolean
        use of moment matching
    '''
    if anti_paths is True:
        sn = npr.standard_normal((M + 1, int(I / 2)))
        sn = np.concatenate((sn, -sn), axis = 1)
    else:
        sn = npr.standard_normal((M + 1, I))
    if mo_match is True:
        sn = (sn - sn.mean()) / sn.std()
    return sn

S0 = 100.
r = 0.05
sigma = 0.25
T = 1.0
I = 50000

def gbm_mcs_stat(K):
    ''' Valuation of European call option in Black-Scholes-Merton by Monte Carlo
    simulation (of index level at maturity)
    
    Parameters
    ==========
    K: float
        (positive) strike price of the option
    
    Returns
    =======
    C0: float
        estimated present value of European call option
    '''
    sn = gen_sn(1, I) # generates random values
    # simulate index level at maturity
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) ** T
                     + sigma * math.sqrt(T) * sn[1])
    # calculate payoff at maturity
    hT = np.maximum(ST - K, 0)
    # calculate MCS estimator
    C0 = math.exp(-r * T) * np.mean(hT)
    return C0

gbm_mcs_stat(K = 105.) # Monte Carlo estimator value for European call option
# Returns: 10.085040975525603

#%% Can use a dynamic simulation approach and allow for European put options as 
# in addition to the call option.  The code also compares option price estimates
# for a call and a put stroke at the same level

M = 50 # Number of time intervals for discretization

def gbm_mcs_dyna(K, option = 'call'):
    ''' Valuation of European options in Black-Scholes-Merton by Monte Carlo
    simulation (of index level paths)
    
    Parameters
    ==========
    K: float
        (positive) strike price of the option
    option : string
        type of the option to be valued ('call', 'put')
    
    Returns
    =======
    C0: float
        estimated present value of European call option
    '''
    dt = T / M
    # simulation of index level paths
    S = np.zeros((M + 1, I))
    S[0] = S0
    sn = gen_sn(M, I)
    for t in range(1, M + 1):
        S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt
               + sigma * math.sqrt(dt) * sn[t])
    # case-based calculation of payoff
    if option == 'call':
        hT = np.maximum(S[-1] - K, 0)
    else:
        hT = np.maximum(K - S[-1], 0)
    # calculation of MCS estimator
    C0 = math.exp(-r * T) * np.mean(hT)
    return C0

gbm_mcs_dyna(K = 110., option = 'call') # 7.971685119016455 -> Estimate for European call

gbm_mcs_dyna(K = 110., option = 'put') # 12.685453606750155 -> Estimate for European put

#%% Comparing the previous performances relative to benchmark value from the
# Black-Scholes valuation formula.  Following code generates respective option 
# values/estimates for a range of strike prices using the analytical option pricing
# formula for European calls found in the module bsm_functions.py

# First, we compare the results from the static simulation approach w/ precise
# analytical values:

from bsm_functions import bsm_call_value

stat_res = []
dyna_res = []
anal_res = []

# ndarray object containing range of strike prices
k_list = np.arange(80., 120.1, 5.)
np.random.seed(100)

for K in k_list:
    # Simulates/calculates and collects option values for all strike prices
    stat_res.append(gbm_mcs_stat(K))
    dyna_res.append(gbm_mcs_dyna(K))
    anal_res.append(bsm_call_value(S0, K, T, r, sigma))

# Transforms list objects to ndarray objects
stat_res = np.array(stat_res)
dyna_res = np.array(dyna_res)
anal_res = np.array(anal_res)

plt.figure(figsize = (10,6))
fix, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize = (10,6))
ax1.plot(k_list, anal_res, 'b', label = 'analytical')
ax1.plot(k_list, stat_res, 'ro', label = 'static')
ax1.set_ylabel('European call option value')
ax1.legend(loc = 0)
ax1.set_ylim(bottom = 0)
wi = 1 # bar width and adjusting alignment
ax2.bar(k_list - wi / 2, (anal_res - stat_res) / anal_res * 100, wi)
ax2.set_xlabel('strike')
ax2.set_ylabel('difference in %')
ax2.set_xlim(left = 75, right = 120)
plt.grid()
plt.suptitle('Analytical Option Values VS Monte Carlo Estimators (Static Simulation)')

#%% Similar picture emerges for dynamic simulation and valuation approach

fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize = (10,6))
ax1.plot(k_list, anal_res, 'b', label = 'analytical')
ax1.plot(k_list, dyna_res, 'ro', label = 'dynamic')
ax1.set_ylabel('European call option value')
ax1.legend(loc = 0)
ax1.set_ylim(bottom = 0)
wi = 1 # bar width and adjusting alignment
ax2.bar(k_list - wi / 2, (anal_res - dyna_res) / anal_res * 100, wi)
ax2.set_xlabel('strike')
ax2.set_ylabel('difference in %')
ax2.set_xlim(left = 75, right = 120)
plt.grid()
plt.suptitle('Analytical Option Values VS Monte Carlo Estimators (Dynamic Simulation)')

# All valuation differences are < 1% (both +ve and -ve).  As a rule of thumb, 
# the quality of Monte Carlo estimator can be controlled for by adjusting the
# number of time intervals M used and/or the number of paths I simulated


# =============================================================================
# American Options
# Since American options are more involved, we need an "optimal stopping" solution
# to come up with a fair value of the option.
# =============================================================================
# The formulation is based on a discrete time grid for use with numerical simulation
# In a sense, it is therefore more ocrrect to speak of an option value given 
# Bermudan exercise (Bermudan options can be exercised on predetermined dates.  
# For the time interval converging to zero length, the value of the Bermudan option
# converges to the one of the American option.

# Using a "Least-Squares Monte Carlo (LSM) algorithm from the paper by Longstaff
# and Schwarz (2001), we can calculate the "continuation value" of the option
# given an index level of S_t = s

def gbm_mcs_amer(K, option = 'call'):
    ''' Valuation of American option in Black-Scholes-Merton by Monte Carlo
    simulation by LSM algorithm
    
    Parameters
    ==========
    K: float
        (positive) strike price of the option
    option: string
        type of the option to be valued ('call', 'put')
        
    Returns
    =======
    C0: float
        estimated present value of American call option
    '''
    dt = T / M
    df = math.exp(-r * dt)
    # simulation of index levels
    S = np.zeros((M + 1, I))
    S[0] = S0
    sn = gen_sn(M, I)
    for t in range(1, M + 1):
        S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt
               + sigma * math.sqrt(dt) * sn[t])
    # case based calculation of payoff
    if option == 'call':
        h = np.maximum(S - K, 0)
    else:
        h = np.maximum(K - S, 0)
    # LSM algorithm
    V = np.copy(h)
    for t in range(M - 1, 0, -1):
        reg = np.polyfit(S[t], V[t + 1] * df, 7)
        C = np.polyval(reg, S[t])
        V[t] = np.where(C > h[t], V[t + 1] *df, h[t])
    # MCS estimator
    C0 = df * np.mean(V[1])
    return C0

gbm_mcs_amer(110., option = 'call') # 7.810714548188455
gbm_mcs_amer(110., option = 'put') # 13.661853293823606

# The European value of an option represents a lower bound to the American option's
# value.  The difference is called the "early exercise premium".  What follows
# compares European and American option values for the same range of strikes as
# before to estimate the early exercise premium, this time with puts (since no
# dividene payments are assumed, there generally is no early exercise premium
# for call options (i.e., no incentive to exercise the option early.))

euro_res = []
amer_res = []

k_list = np.arange(80., 120.1, 5.)

for K in k_list:
    euro_res.append(gbm_mcs_dyna(K, 'put'))
    amer_res.append(gbm_mcs_amer(K, 'put'))
    
euro_res = np.array(euro_res)
amer_res = np.array(amer_res)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize = (10, 6))
ax1.plot(k_list, euro_res, 'b', label = 'European put')
ax1.plot(k_list, amer_res, 'ro', label = 'American put')
ax1.set_ylabel('call option value')
ax1.legend(loc = 0)
wi = 1.0
ax2.bar(k_list - wi / 2, (amer_res - euro_res) / euro_res * 100, wi)
ax2.set_xlabel('strike')
ax2.set_ylabel('early exercise premium in %')
ax2.set_xlim(left = 75, right = 125)