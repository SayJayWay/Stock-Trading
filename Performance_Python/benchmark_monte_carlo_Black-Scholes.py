#%%        
# =============================================================================
# Monte Carlo Simulation based on  Black-Scholes Model for European call option
# =============================================================================
# Pure Python

import math
import numpy as np

S0 = 36. # Initial value of risky asset
T = 1.0 # Time horizon for the binomial tree simulation
r = 0.06 # Constant short rate
sigma = 0.2 # Constant volatility factor
M = 100 # Number of time intervals for discretization
I = 50000 # Number of paths to be simulated

def mcs_simulation_py(p):
    M, I = p
    dt = T / M
    S = np.zeros((M + 1, I))
    S[0] = S0
    rn = np.random.standard_normal(S.shape) # Random numbers, drawn in a single vectorized step
    for t in range(1, M + 1):
        for i in range(I): # Nested loop implementing simulation based on Euler scheme
            S[t, i] = S[t-1, i] * math.exp((r - sigma ** 2 / 2) * dt +
                                 sigma * math.sqrt(dt) * rn[t, i])
    return S

%time S = mcs_simulation_py((M,I)) # 6.27

S[-1].mean() # Mean end-of-period value based on simulation
S0 * math.exp(r * T) # Theoretically expected end-of-period value
K = 40. # Strike price of European put option
C0 = math.exp(-r * T) * np.maximum(K - S[-1], 0).mean() # Monte Carlo estimator for put option

#%% Numpy method -> Will still have to loop over time intervals, but the other
#   dimension is handled by vectorized code over all paths

def mcs_simulation_np(p):
    M, I = p
    dt = T / M
    S = np.zeros((M + 1, I))
    S[0] = S0
    rn = np.random.standard_normal(S.shape) # Random numbers, drawn in a single vectorized step
    # This for-loop was changed (did not need nested loop)
    for t in range(1, M + 1):
        S[t] = S[t-1] * np.exp((r - sigma ** 2 / 2) * dt +
                                 sigma * math.sqrt(dt) * rn[t])
    return S

%time S = mcs_simulation_np((M,I)) # 172ms

S[-1].mean()
%timeit S = mcs_simulation_np((M,I))
# 214 ms ± 10.4 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

#%% Numba method

import numba

mcs_simulation_nb = numba.jit(mcs_simulation_py)

%time S = mcs_simulation_nb((M, I))# 425ms
%time S = mcs_simulation_nb((M, I))# 170ms

print(S[-1].mean())

C0 = math.exp(-r * T) * np.maximum(K - S[-1], 0).mean()

%timeit S = mcs_simulation_nb((M, I))
# 168 ms ± 1.83 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

#%% Cython method -> Not surprisingly, effort required to speed up is higher
#   This time around, seems like cython is slower than numpy and numba as, along
#   with other factors, some time is needed to transform the simulation results
#   to an ndarray object:

%%cython
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, sqrt
cdef float S0 = 36.
cdef float T = 1.0
cdef float r = 0.06
cdef float sigma = 0.2
@cython.boundscheck(False)
@cython.wraparound(False)
def mcs_simulation_cy(p):
    cdef int M, I
    M, I = p
    cdef int t, i
    cdef float dt = T / M
    cdef double[:, :] S = np.zeros((M + 1, I))
    cdef double[:, :] rn = np.random.standard_normal((M + 1, I))
    S[0] = S0
    for t in range(1, M + 1):
        for i in range(I):
            S[t, i] = S[t-1, i] * exp((r - sigma ** 2 / 2) * dt +
                                             sigma * sqrt(dt) * rn[t, i])
    return np.array(S)

%time S = mcs_simulation_cy((M, I)) # 198ms

print(S[-1].mean())

%timeit S = mcs_simulation_cy((M, I))
# 238 ms ± 10.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

#%% Multiprocessing method works well with Monte Carlo simulations.  One approach
#   is to parallelize an ex of 100,000 paths into 10 processes simulating 10,000
#   paths each.  Another would be to parallelize the simulation of 100,000 paths into
#   multiple processes, each simulating a different financial instrument.

#   Following code will divide total number of paths to be simulated I into smaller
#   chunks of size I / p with p > 0.  After all the single tasks are finished, results
#   are combined into a single ndarray object via np.hstack().  In this specific
#   example, there is no speedup.

import multiprocessing as mp
import numpy as np

def main():  
    pool = mp.Pool(processes = 4) # Pool object for parallelization
    
    p = 20 # Number of chunks in which simulation is divided
    
    %timeit S = np.hstack(pool.map(mcs_simulation_np, p * [(M, int(I / p))]))   
    %timeit S = np.hstack(pool.map(mcs_simulation_nb, p * [(M, int(I / p))]))   
    %timeit S = np.hstack(pool.map(mcs_simulation_cy, p * [(M, int(I / p))]))   

if __name__ == '__main__':
    main()
    
# This hangs my computer...Could not figure out why..
   
#%%
# =============================================================================
# Recursive pandas.  This is necessary to know as certain recursive algorithms
# are hard or impossible to vectorize, which leaves the user with slow Python
# loops on DataFrame objects.
# =============================================================================
# Calculating EMA
# EMA_0 = S0
# EMA_t = alpha* S_t + (1-alpha) * EMA_(t-1), where t E {1,...,T}

#%% First Python

import pandas as pd
import numpy as np

sym = 'Adj Close'
data = pd.DataFrame(pd.read_csv('../stock_dfs/AAPL.csv', index_col = 0,
                                parse_dates = True)[sym]).dropna()

alpha = 0.25
data['EWMA'] = data[sym]

#%%time
#for t in zip(data.index, data.index[1:]):
#    data.loc[t[1], 'EWMA'] = (alpha * data.loc[t[1], sym] +
#            (1 - alpha) * data.loc[t[0], 'EWMA'])
# 176ms

data[data.index > '2017-01-01'].plot(figsize=(10,6))

#%% More general Python function that can be applied directly on the column or
#   raw financial times series data in form of ndarray object

def ewma_py(x, alpha):
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1-alpha) * y[i-1]
    return y

# Applying function to series object directly (i.e. the column)
%time data['EWMA_PY'] = ewma_py(data[sym], alpha) # 8.04ms

# Applying function to ndarray object containing raw data
%time data['EWMA_PY'] = ewma_py(data[sym].values, alpha) # 0ns

# We can already see it is 20x-100x faster

#%% Numba version

import numba

ewma_nb = numba.jit(ewma_py)

%time data['EWMA_nb'] = ewma_nb(data[sym], alpha) # 500ms
    
%timeit data['EWMA_nb'] = ewma_nb(data[sym], alpha)
# 9.42 ms ± 384 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    
%time data['EWMA_nb'] = ewma_nb(data[sym].values, alpha) # 79.9ms
    
%timeit data['EWMA_nb'] = ewma_nb(data[sym].values, alpha)
# 83 µs ± 2.64 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

#%% Cython also improves, but not as fast as Numba

%%cython
import numpy as np
cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)
def ewma_cy(double[:] x, float alpha):
    cdef int i
    cdef double[:] y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1-alpha) * y[i - 1]
    return y

%time data['EMA_CY'] = ewma_cy(data[sym].values, alpha) # 0ns

%timeit data['EMA_CY'] = ewma_cy(data[sym].values, alpha)
# 240 µs ± 9.83 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)