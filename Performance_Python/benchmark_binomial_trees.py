# =============================================================================
# Binomial Trees
# Bionomial option pricing model is a popular numerical method to value options
# =============================================================================

#%% Black-Scholes model set up.  This is used to estimate pricing of a European
#   option (European options can not be striked before the expiration date). 
#   In this model, there is a risky asset (index/stock) and riskless asset (bond)
#   Relevant time interval from today until maturity of option is divided in general
#   into equidistant subintervals of length dt.  Given an index level at a time s
#   of S_s, the index level at t = s + dt is given by S_t = S_s * m, where m is chosen
#   randomly from {u, d} with 0 < d < e^(r*dt) < u = e^(sigma*sqrt(dt)) as well as
#   u = 1/d*r is the constant, riskless short rate.

#%% Python implementation that creates a recombining tree based on some fixed
#   numerical parameters for the model

import math
import numpy as np

S0 = 36. # Initial value of risky asset
T = 1.0 # Time horizon for the binomial tree simulation
r = 0.06 # Constant short rate
sigma = 0.2 # Constant volatility factor

def simulate_tree(M):
    dt = T / M # Length of time intervals
    u = math.exp(sigma * math.sqrt(dt)) # Factors for upward and downward movements
    d = 1 / u # Factors for upward and downward movements
    S = np.zeros((M + 1, M + 1))
    S[0 ,0] = S0
    z = 1
    for t in range(1, M + 1):
        for i in range(z):
            S[i, t] = S[i, t-1] * u
            S[i+1, t] = S[i, t-1] * d
        z += 1
    return S

# Contrary to typical tree plots, an upward movement is represented in the ndarray
# object as a sideways movement, which decreases the ndarray size considerably:

np.set_printoptions(formatter={'float': lambda x: '%6.2f' % x})

simulate_tree(500) #%time 94.3ms

#%% Binomial tree can be created with NumPy

def simulate_tree_np(M):
    dt = T / M
    up = np.arange(M + 1)
    up = np.resize(up, (M + 1, M + 1)) # Gross upward movements
    down = up.transpose() * 2 # Gross downward movements
    
    # up-down returns net upward (+ve) and downward (-ve) movements
    S = S0 * np.exp(sigma * math.sqrt(dt) * (up-down))
    return S

simulate_tree_np(500) #%time 14.2ms

#%% With Numba

import numba

simulate_tree_nb = numba.jit(simulate_tree_np)

simulate_tree_nb(500)#%time 141 ms
simulate_tree_nb(500)#%time 4.32 ms

#%% With Cython

%%cython -a
import numpy as np
cimport cython
from libc.math cimport exp, sqrt
cdef float S0 = 36.
cdef float T = 1.0
cdef float r = 0.06
cdef float sigma = 0.2
def simulate_tree_cy(int M):
    cdef int z, t, i
    cdef float dt, u, d
    cdef float[:, :] S = np.zeros((M + 1, M + 1),
                                  dtype = np.float32)
    dt = T / M
    u = exp(sigma * sqrt(dt))
    d = 1 / u
    S[0, 0] = S0
    z = 1
    for t in range(1, M + 1):
        for i in range(z):
            S[i, t] = S[i, t-1] * u
            S[i + 1, t] = S[i, t-1] * d
        z += 1

# %time simulate_tree_cy(500) 645us
