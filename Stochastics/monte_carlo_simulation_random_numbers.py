#! Monte Carlo Simulation using the  Black-Scholes-Merton equation

import math
import numpy as np
import numpy.random as npr
from pylab import plt, mpl

S0 = 100 # Initial index price
r = 0.05 # Constant riskless short rate
sigma = 0.25 # Constant volatility factor
T = 2.0 # Horizon in year fractions
I = 10000 # Number of simulations

# Simulation via a vectorized expression; the discretization scheme makes use
# of npr.standard_normal() function
ST1 = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * math.sqrt(T) *
                  npr.standard_normal(I))

plt.figure(figsize = (10,6))
plt.hist(ST1, bins = 50)
plt.xlabel('index level')
plt.ylabel('frequency')
plt.title('Standard Normal dist')

# The graph suggests that the distribution of the random variable in the Black-
# Scholes-Merton equation is log-normal.  We can therefore try and use the npr.lognormal()
# function to directly derive values for the random variable.  With this formula,
# one has to provide hte mean and std to the function

ST2 = S0 * npr.lognormal((r - 0.5 * sigma ** 2) * T,
                         sigma * math.sqrt(T), size = I)

plt.figure(figsize = (10,6))
plt.hist(ST2, bins = 50)
plt.xlabel('index level')
plt.ylabel('frequency')
plt.title('Lognormal dist')

#%% Comparing the two graphs
# First and 2nd figure indeed look similar...We can compare them using scipy.stats
# subpackage and the helpfer function print_statistics()

import scipy.stats as scs

def print_statistics(a1, a2):
    ''' Prints selected statistics.
    
    Parameters
    ==========
    a1, a2: ndarray objects
            results objects from simulation
    '''
    sta1 = scs.describe(a1) # Gives back important stats for the function
    sta2 = scs.describe(a2)
    print('%14s %14s %14s' % ('statistics', 'data set 1', 'data set 2'))
    print(45 * "-")
    print('%14s %14.3f %14.3f' % ('size', sta1[0], sta2[0]))
    print('%14s %14.3f %14.3f' % ('min', sta1[1][0], sta2[1][0]))
    print('%14s %14.3f %14.3f' % ('max', sta1[1][1], sta2[1][1]))
    print('%14s %14.3f %14.3f' % ('mean', sta1[2], sta2[2]))
    print('%14s %14.3f %14.3f' % ('std', np.sqrt(sta2[3]), np.sqrt(sta2[3])))
    
    # Skew is measure of symmetry or, more precisely, lack of asymettry
    print('%14s %14.3f %14.3f' % ('skew', sta1[4], sta2[4])) 
    
    # Kurtosis is a measure of whether the data are heavy-tailed or light-tailed 
    # relative to a normal distribution. data sets with high kurtosis tend to have
    # heavy tails, or outliers.
    print('%14s %14.3f %14.3f' % ('kurtosis', sta1[5], sta2[5]))
    

print_statistics(ST1, ST2)

# Results show stats are quite similar.  Differences is due to 'sampling error'
# within the simulation.  Another error that can be seen when simulating continuous
# stochastic processes is discretization error (which plays no role here due to
# the nature of thte simulation approach).

#%%
# =============================================================================
# Stochastic Processes
# A stochastic process is a sequence of random variables.  In that sense, one 
# should expetc something similar to a sequence of repeated simulations of a 
# random variable when simulating a process.  This is mainly true, except for the
# fact that the draws are typically not independent but rather depend on the 
# result(s) of the previous draw(s).  In finance, stochastic processes tend to 
# exhibit the Markov property, which mainly says that tomorrow's value of the 
# process only depends on today's state of the process, and not any other more
# 'historic' state of even the whole path history.  The process is then also called
# 'memoryless'
# =============================================================================

# Can now convert Black-Scholes model into a dynamic form, as described by the
# stochastic differential equation (SDR) called a geometric Brownian motion.
# i.e. all we do is switch all time values to dt

I = 10000 # Number of paths to be simulated
M = 50 # Number of time intervals for discretization
dt = T / M # Step size in year fractions
S = np.zeros((M + 1, I)) # 2D ndarray object for index levels
S[0] = S0 # Initial value (100) for initial point in time t = 0 

# Simulation via vectorized expression; loop is over points in time starting at
# t = 1 and ending at t = T
for t in range(1, M + 1):
    S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt +
            sigma * math.sqrt(dt) * npr.standard_normal(I))
    
plt.figure(figsize = (10,6))
plt.hist(S[-1], bins = 50)
plt.xlabel('index level')
plt.ylabel('frequency')
plt.title('Dynamic Black Scholes')

print('-' * 45)
print('Black Scholes Dynamic')
print('-' * 45)
print_statistics(S[-1], ST2)

# Displaying the first 10 simulated paths
plt.figure(figsize = (10,6))
plt.plot(S[:, :10], lw = 1.5)
plt.xlabel('time')
plt.ylabel('index level')
plt.title('First 10 paths from Dynamic Black Scholes')

# Visualizing the first 10 paths allows us to value options with exercise or
# options whose payoff is path-dependent.  One gets the full dynamic picture over
# time so to say.

#%% Square-root diffusion
# Another important class of financial processes is mean-reverting processes, which
# are used to model short rates or volatility processes.  Popular and widely used
# model is the 'squart-root diffusion':
#   dx_t = k(theta-x_t)dt + sigma*sqrt(x_t)dZ_t
# where x_t = Process level at date t (Chi-squared distributed)
#       k = Mean-reversion factor
#       theta = Long-term mean of the process
#       sigma = Constant volatility parameter
#       Z_t = Standard Brownian motion

# Many financial models can be discretized and approximated using normal distribution
# (i.e., Euler discretization scheme).  While Euler scheme is exact for the 
# geometric Brownian motion, it is biased for majority of other stochastic processes.
# There might be better discretization schemes, however, Euler scheme might be
# desirable for numerical and/or computational reasons.

# Discretizing this equations leads to

# x~_t = x~_s + k(theta - x~+_s)deltat + sigma*sqrt(s~+_s)*sqrt(deltat)*z_t
# X_t = x~+_t
# where s = t - deltat
#       x+ == max(x,0)

# Square-root diffusion has convenient and realistic characteristic that the 
# values of x_t remain strictly positive.  When discretizing it by an Euler
# scheme, negative values cannot be excluded.  this is the reason why one works
# always with the positive version of the originally simmulated process. In
# the simulation code, one therefore needs two ndarray objects instead of only one.

x0 = 0.05 # Initial Value (for short-rate)
kappa = 3.0 # Mean reversion factor
theta = 0.02 # Long term mean value
sigma = 0.1 # Volatility factor
I = 1000
M = 50
dt = T / M

def srd_euler():
    xh = np.zeros((M + 1, I ))
    x = np.zeros_like(xh) # Returns an array of zeros with the same shape as given array
    xh[0] = x0
    x[0] = x0
    for t in range(1, M + 1):
        xh[t] = (xh[t-1] +
                  kappa * (theta - np.maximum(xh[t-1], 0)) * dt +
                  sigma * np.sqrt(np.maximum(xh[t-1], 0)) *
                  math.sqrt(dt) *
                  npr.standard_normal(I)) # this is z_t
    # This will only take positive values (compares two arrays and takes larger
    # value...Since comparing w/ 0, it will take 0 over a negative value)
    x = np.maximum(xh, 0) 
    return x # Returns matrix of positive values 

x1 = srd_euler()

plt.figure(figsize = (10,6))
plt.hist(x1[-2], bins = 50)
plt.xlabel('value')
plt.ylabel('frequency')
plt.title('Dynamically simulated square-root diffusion at maturity (Euler scheme)')

# Plotting first 10 simulated paths
plt.figure(figsize = (10,6))
plt.plot(x1[:, :10], lw = 1.5)
plt.xlabel('time')
plt.ylabel('index level')
plt.title('Dynamically simulated square-root diffusion paths (Euler scheme)')
# Notice the resulting negative average drift (due to initial value being greater
# than the average mean -> i.e. x_0 > theta) and the convergence to theta = 0.02

#%% Can use exact discretization scheme for square-root diffusion based on the 
# non central chi-square distribution (can google this eq'n)

def srd_exact():
    x = np.zeros((M + 1, I))
    x[0] = x0
    for t in range(1, M + 1):
        df = 4 * theta * kappa / sigma ** 2
        c = (sigma ** 2 * (1 - np.exp(-kappa * dt))) / (4 * kappa)
        nc = np.exp(-kappa * dt) / c * x[t - 1]
        x[t] = c * npr.noncentral_chisquare(df, nc, size = I)
    return x

x2 = srd_exact()

plt.figure(figsize = (10,6))
plt.hist(x2[-1], bins = 50)
plt.xlabel('value')
plt.ylabel('frequency')
plt.title('Dynamically simulated square-root diffusion at maturity (exact scheme)')

plt.figure(figsize = (10,6))
plt.plot(x2[:, :10], lw = 1.5)
plt.xlabel('time')
plt.ylabel('index level')
plt.title('Dynamically simulated square-root diffusion paths (exact scheme)')

# Comparing Euler vs exact scheme statistics
print('-' * 45, '\n', 'Euler VS Exact scheme\n', '-' * 45)
print_statistics(x1[-1], x2[-1])
# Can see that Euler scheme performs quite well VS desired statistical properties

I = 250000
#%time x1 = srd_euler() # 945ms
#%time x2 = srd_exact() # 1.53s

# However, we can see a major difference in execution speeds since sampling from
# the noncentral chi-square distribution is more computationally demanding than
# from the standard normal distribution

#%% Stochastic volatility
# Since volatility is not constant nor deterministic (it is stochastic), stochastic
# volatility is extremely important.  One of the most popular models that fall
# into that category is that of Heston (1993).

# Rho represents instantaneous correlation b/w two standard Brownian motions Z1_t, Z2_t
# This allows us to account for stylized fact called leverage effect, which states
# that volatility goes up in times of stress (declining markets) and goes down
# in bull markets

# To account for correlation b/w to stochastic processes, one needs to determine
# the Cholesky decomposition of the correlation matrix.  Matrix decompositions 
# (or matrix factorization) is a factorization of a matrix into a product of matrices
# Cholesky decomposition is one type of this that results in a lower traingular
# matrix with real and positive diagonal entries and a conjugate transpose of this
# matrix.   https://www.geeksforgeeks.org/cholesky-decomposition-matrix-decomposition/
# Cholesky decomposition is to break down larger Hermitian matrices into smaller
# 3x3 matrices.  This is much easier for computation and will speed up the process.

#%% Example of Cholesky decompositon
S0 = 100
r = 0.05
v0 = 0.1 # Initial volatility value
kappa = 3.0
theta = 0.25
sigma = 0.1
rho = 0.6 # Fixed correlation between two Brownian motions
T = 1.0

corr_mat = np.zeros((2,2))
corr_mat[0, :] = [1.0, rho]
corr_mat[1, :] = [rho, 1.0]
cho_mat = np.linalg.cholesky(corr_mat) # Cholesky decomposition
cho_mat

#%% Stochastic Volatility with Euler scheme

M = 50
I = 10000
dt = T / M

ran_num = npr.standard_normal((2, M + 1, I)) # 3D random number data set

v = np.zeros_like(ran_num[0])
vh = np.zeros_like(v)

v[0] = v0
vh[0] = v0

for t in range(1, M + 1):
    # Picks out relevant random number subset and transforms it via Cholesky matrix
    ran = np.dot(cho_mat, ran_num[:, t, :])
    vh[t] = (vh[t - 1] +
             kappa * (theta - np.maximum(vh[t - 1], 0)) * dt +
             sigma * np.sqrt(np.maximum(vh[t-1], 0)) *
             math.sqrt(dt) * ran[1]) # Simulates paths based on Euler scheme
    
v = np.maximum(vh , 0) # Volatility

S = np.zeros_like(ran_num[0]) # Setting up index level matrix

S[0] = 50
for t in range(1, M + 1):
    ran = np.dot(cho_mat, ran_num[:, t, :])
    S[t] = S[t - 1] * np.exp((r - 0.5 * v[t]) * dt +
             np.sqrt(v[t]) * ran[0] * np.sqrt(dt))
    
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 6))
ax1.hist(S[-1], bins = 50)
ax1.set_xlabel('index level')
ax1.set_ylabel('frequency')
ax1.set_title('Distribution of Index level at maturity')
ax2.hist(v[-1], bins = 50)
ax2.set_xlabel('volatility')
ax2.set_title('Dynamically simulated stochastic volatility process at maturity')

print_statistics(S[-1], v[-1])
# Stats at maturity (i.e. [-1]) for both data sets reveal a high maximum value
# for the index level process.  This is much higher than a geometric Brownian
# motion w/ constant volatility could ever climb.

fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize = (10,6))

ax1.plot(S[:, :10], lw= 1.5)
ax1.set_ylabel('index level')
ax2.plot(v[:, :10], lw = 1.5)
ax2.set_xlabel('time')
ax2.set_ylabel('volatility')
fig.suptitle('Dynamically Simulated Stochastic Volatility First 10 Process Paths')

# Can see that volatility drifts positively towards theta = 0.25 as expected.

#%% Jump Diffusion
# Another stylized (empirical) fact is the existence of "jumps" in asset prices
# and, for example, volatility.  Merton jump diffusion model (which is an extension
# of Black-Scholes setup) generates jumps with log-normal distribution and models 
# sudden asset price movements (both up and down) by adding the jump diffusion 
# parameters with the Poisson process Pt.

#% Stochastic differential equation for Merton jump diffusion model
# dS_t = (r-r_j) * S_t*dt + sigma*S_t*dZ_t + J_t*S_t*dN_t
#   where S_t = Index level at date t
#         t = Constant riskless short rate
#         r_j == lambda * (e^(mu_j + delta^2/2)-1) = Drift correction for jump to maintain risk neutrality
#         sigma = Constant volatility of S
#         Z_t = Standard Brownian motion
#         J_t = Jump at date t w/ distribution..
#               ..log(1 + J_t) ~= N(log(1+mu_J) - delta^2/2, delta^2) with...
#               ..N as the cumulative distribution function of a standard normal random variable
#         N_t = Poisson process with intensity lambda

#% Euler discretization for Merton jump diffusion model 
# S_t = S_(t-delta_t)*(e^((r-r_j-sigma^2/2)*delta_t + sigma(sqrt(delta_t)*z_t^1)) +
#            (e^(mu_j + delta*z_t^2)-1)*y_t)
# With this discretization scheme, we can do some numerical parameterization

S0 = 100.
r = 0.05
sigma = 0.2
lamb = 0.75 # Jump intensity
mu = -0.6 # Mean jump size
delta = 0.25 # Jump volatility
rj = lamb * (math.exp(mu + 0.5 * delta ** 2) -1) # Drift correction

# This time, 3 sets of random numbers are needed.

S = np.zeros((M + 1, I))
S[0] = S0
sn1 = npr.standard_normal((M + 1, I)) # Standard normally distributed numbers
sn2 = npr.standard_normal((M + 1, I)) # Standard normally distributed numbers
poi = npr.poisson(lamb * dt, (M + 1, I)) # Poisson distributed numbers
for t in range(1, M + 1, 1):
    S[t] = S[t - 1] * (np.exp((r - rj - 0.5 * sigma ** 2) * dt +
           sigma * math.sqrt(dt) * sn1[t]) +
           (np.exp(mu + delta * sn2[t]) -1) *
           poi[t]) # Simulation based on exact Euler scheme
    S[t] = np.maximum(S[t], 0)
    
plt.figure(figsize = (10, 6))
plt.hist(S[-1], bins = 50)
plt.xlabel('value')
plt.ylabel('frequency')
plt.title('Dynamically simulated jump diffusion process at maturity')
# Notice the second peak (bimodal frequency distribution), which is due to the jumps

# Negative jumps can also be spotted in the first 10 simulated index level paths
plt.figure(figsize = (10,6))
plt.plot(S[:, :10], lw = 1.5)
plt.xlabel('time')
plt.ylabel('index level')
plt.title('Dynamically simulated jump diffusion process paths')