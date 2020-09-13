# =============================================================================
# Value-at-risk (VaR)
# VaR is a number denoted in currency units (i.e. USD, JPY, etc) indicating a 
# loss (of a portfolio, single position, etc) that is not exceeded with some
# confidence level (probability) over a given period of time.
# =============================================================================
# Example:
# Consider a stock position worth 1,000,000 USD that has a VaR of 50,000 USD at
# a confidence level of 99% over a time period of 30 days (one month).  This says
# that with 99% probability, the loss to be expected over a period of 30 days will
# not EXCEED 50,000 USD. However, it does not say anything about the size of loss
# once a loss beyond that amount occurs (i.e. if the max loss is 100,000 USD or
# 500,000 USD).  All it says is that there is a 1% chance that a loss of a
# MINUMUM of 50,000USD or higher will occur.

# Assume Black-Scholes setup and consider a future date T = 30/365 (a period of
# 30 days)

import math
import numpy as np
import numpy.random as npr
from pylab import mpl, plt
import scipy.stats as scs

S0 = 100
r = 0.05
sigma = 0.25
T = 30 / 365.
I = 10000


# Simulates end-of-period values for geometric Brownian motion
ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T +
                 sigma * np.sqrt(T) * npr.standard_normal(I))

# Calculates absolute profits and losses per simulatin run and sorts the values
R_gbm = np.sort(ST - S0)

plt.figure(figsize = (10, 6))
plt.hist(R_gbm, bins = 50)
plt.xlabel('absolute return')
plt.ylabel('frequency')
plt.title('Absolute profits and losses from simulation (geometric Brownian Motion')

# The reason we sort the results is so we can use scs.scoreatpercentile() to
# calculate the VaR at a specific confidence level by defining the list object
# percs. 0.1 translates to confidence level of 100% - 0.1% = 99.9%

percs = [0.01, 0.1, 1., 2.5, 5.0, 10.0]
var = scs.scoreatpercentile(R_gbm, percs)
print('%16s %16s' % ('Confidence Level', 'Value-at-Risk'))
print(33 * '-')
for pair in zip(percs, var):
    print('%16.2f %16.3f' % (100 - pair[0], -pair[1]))
    
#Confidence Level    Value-at-Risk
#---------------------------------
#           99.99           22.175
#           99.90           19.642
#           99.00           14.996
#           97.50           12.941
#           95.00           10.987
#           90.00            8.494
    
#%% With Dynamic Jump Diffusion
# If we plot this with the dynamic jump diffusion setup with a negative mean,
# we see a bimodal distribution.  From a normal distribution point of view, one
# sees a pronounced left fat tail

M = 50
lamb = 0.75 # Jump intensity
mu = -0.6 # Mean jump size
delta = 0.25 # Jump volatility

dt = 30. / 365 / M
rj = lamb * (math.exp(mu + 0.5 * delta ** 2) - 1) # Drift correction

S = np.zeros((M + 1, I))
S[0] = S0
sn1 = npr.standard_normal((M + 1, I))
sn2 = npr.standard_normal((M + 1, I))
poi = npr.poisson(lamb * dt, (M + 1, I))
for t in range(1, M + 1, 1):
    S[t] = S[t - 1] * (np.exp((r - rj - 0.5 * sigma ** 2) * dt
                       + sigma * math.sqrt(dt) * sn1[t])
                       + (np.exp(mu + delta * sn2[t]) - 1)
                       * poi[t])
    S[t] = np.maximum(S[t], 0)
    
R_jd = np.sort(S[-1] - S0)

plt.figure(figsize = (10, 6))
plt.hist(R_jd, bins = 50)
plt.xlabel('absolute return')
plt.ylabel('frequency')

percs = [0.01, 0.1, 1., 2.5, 5.0, 10.0]
var = scs.scoreatpercentile(R_jd, percs)
print('%16s %16s' % ('Confidence Level', 'Value-at-Risk'))
print(33 * '-')
for pair in zip(percs, var):
    print('%16.2f %16.3f' % (100 - pair[0], -pair[1]))
    
#Confidence Level    Value-at-Risk
#---------------------------------
#           99.99           78.042
#           99.90           71.326
#           99.00           56.575
#           97.50           45.970
#           95.00           25.774
#           90.00            8.475
    
# Can see that the 99.9% levels is more than 3x as high but the 90% level
# is almost identical to that of gemoetric Brownian motion
# This illustrates the problem of capturing the tail risk so often encountered
# in financiel markets by standard VaR measure
    
# To further illustrate this point, can compare both cases graphically

percs = list(np.arange(0.0, 10.1, 0.1))
gbm_var = scs.scoreatpercentile(R_gbm, percs)
jd_var = scs.scoreatpercentile(R_jd, percs)

plt.figure(figsize = (10, 6))
plt.plot(percs, gbm_var, 'b', lw = 1.5, label = 'GBM')
plt.plot(percs, jd_var, 'r', lw = 1.5, label = 'JD')
plt.legend(loc = 4)
plt.xlabel('100 - confidence level [%]')
plt.ylabel('value-at-risk')
plt.ylim(ymax = 0.0)

# =============================================================================
# Credit Valuation Adjustments
# Another risk measure is the credit value-at-risk (CVaR) and the credit valuation
# adjustment (CVA), which is derived from CVaR.  CVaR is the measure of risk
# resulting from the possibility that a counterparty might not be able to honor
# its obligations (for ex if the counterparty goes bankrupt).  In such a case,
# there are two main assumptions to be made: the "probability by default" and the
# (average) "loss level"
# =============================================================================

# Consider a simple case of Black-Scholes model with a fixed (average) loss
# level L and a fixed probability p of default (per year) of a counterparty.
# Using Poisson distribution, default scenarios are generated as follows, taking
# into account that a default can only occur once.

S0 = 100.
r = 0.05
sigma = 0.2
T = 1.
I = 100000

ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T
                 + sigma * np.sqrt(T) * npr.standard_normal(I))

L = 0.5 # Loss level
p = 0.01 # Probability of default
D = npr.poisson(p * T, I) # Simulates default events (how many times event will
                          # occur at least T times with probability p)
D = np.where(D > 1, 1, D) # Limits defaults to one such event

math.exp(-r * T) * np.mean(ST) # Discounted average simulated value of the asset at T
# Returns: 99.89549114468717

# CVaR as the discounted avg of the future losses in the case of a default
CVaR = math.exp(-r * T) * np.mean(L * D * ST) # 0.5051720821981571

# Discounted avg simulated value of asset at T, adjusted for simulated losses from default
S0_CVA = math.exp(-r * T) * np.mean((1 - L * D) * ST) # 99.39031906248903

# Current price of asset adjusted by simulated CVaR
S0_adj = S0 - CVaR # 99.49482791780184

# Number of defaults
(D>0).sum() # 1002

# Number of losses due to defaults (should be same)
np.count_nonzero(L * D * ST) # 1002

# We can see that there are ~1000 losses due to credit risk, which is expected
# given 1% probability of 100,000 simulated paths.

plt.figure(figsize = (10, 6))
plt.hist(L * D * ST, bins = 50)
plt.xlabel('loss')
plt.ylabel('frequency')
plt.ylim(ymax = 175)
plt.title('Losses due to risk-neutrally expected default (stock)')

#%%
# Consider now a European call option, with value of 10.4 units at strike of 100.
# CVaR is ~5 cents given same assumptions with regard to probability of default
# and loss level.

K = 100.
hT = np.maximum(ST -K, 0)
C0 = math.exp(-r * T) * np.mean(hT) # Monte Carlo estimator value for Euro call option

# CVaR as discounted avg of the future losses in the case of a default
CVaR = math.exp(-r * T) * np.mean(L * D * hT)

# Monte Carlo estimator value for Euro call option, adjusted for simulated losses
# from default
C0_CVA = math.exp(-r * T) * np.mean((1 - L * D) * hT) # 10.41088526471217

# Below can see that although there are ~1000 defaults, we only loss abount 500
# of the times due to default.  This results from the fact that the payoff of
# the option at maturity has a high probability of being zero.

np.count_nonzero(L * D * hT) # 541 -> Losses due to default
np.count_nonzero(D) # 948 -> Number of defaults
I - np.count_nonzero(hT) # 44353 -> Cases for which option expires worthless

plt.figure(figsize = (10, 6))
plt.hist(L * D * hT, bins = 50)
plt.xlabel('loss')
plt.ylabel('frequency')
plt.ylim(ymax = 350)
plt.title('Losses due to risk-neutrally expected default (call option)')