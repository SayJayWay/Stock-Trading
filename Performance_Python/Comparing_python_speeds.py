
# =============================================================================
# Normal for loop
# =============================================================================

import random

def average_py(n):
    s = 0
    for i in range(n):
        s += random.random()
    return s / n

n = 10000000

#average_py(n) # %time ~1.06s
#average_py(n) # %timeit ~897 ms ± 18.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
#sum([random.random() for _ in range(n)]) / n # %time ~1.31s

# =============================================================================
# Using Numpy
# =============================================================================

import numpy as np

def average_np(n):
    s = np.random.random(n)
    return s.mean()

#average_np(n) # %time ~115ms
#average_np(n) # %timeit ~110 ms ± 919 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

# Although numpy is faster by almost a factor of 10, it requires significantly higher
# memory usage.  this is because NumPy attains speed by preallocating data that can
# be processed in the compiled layer.  As a consequence, we can not work with 
# 'streamed' data.
    
# =============================================================================
# Using Numba
# =============================================================================

# numba allows dynamic compilin gof pure Python code by use of LLVM
    
import numba

average_nb = numba.jit(average_py) # This creates the numba function

#average_nb(n) # %time ~94.2ms
#average_nb(n) # %time ~52.3ms
#average_nb(n) # %timeit ~50.9 ms ± 476 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

# Compiling happens during the first runtime, which will lead to some overhead,
# however, from the second execustion (with the same input data types), the
# execution becomes faster

# Numba preservers memory efficiency of original loop-based implementation.
# While numba is generally better, there are many use cases in which it is not 
# suited and for which perofrmance gains are hardly observed or even impossible
# to achieve

# =============================================================================
# Using CPython (Needs to be run in console, not script)
# =============================================================================
# CPython allows one to statically compile code, however, the application is not
# as simple as with Numba since the code generally needs to be changed to see
# significant speed improvements


%load_ext cython

%%cython -a
import random
def average_cy1(int n):
    cdef int i
    cdef float s = 0
    for i in range(n):
        s += random.random()
    return s / n

average_cy1(n) #time ~471ms
average_cy1(n) #timeit 475 ms ± 2.88 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# Some speed up is observed (compared to the for loop), but it is not nearly as 
# fast as the NumPy version...Even more optimization needed to beat Numba

%%cython 
from libc.stdlib cimport rand # imports random number generator from C
cdef extern from 'limits.h': # Imports constant value for the scaling of random numbers
    int INT_MAX
cdef int i
cdef float rn
for i in range(5):
    rn = rand() / INT_MAX # Adds uniformly distributed ran nums from (0,1), after scaling
    print(rn)
    
%%cython -a
from libc.stdlib cimport rand
cdef extern from 'limits.h':
    int INT_MAX
def average_cy2(int n):
    cdef int i
    cdef float s = 0
    for i in range(n):
        s += rand() / INT_MAX
    return s / n

# average_cy2(n) # %time ~191 ms
# average_cy2(n) # %timeit 195 ms ± 1.45 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    
# This was supposed to be faster and more optimized but was not..
# Cython also preserves the memory efficiency of the original loop-based implementation