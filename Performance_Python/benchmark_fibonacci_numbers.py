# -*- coding: utf-8 -*-

# =============================================================================
# Fibonacci Numbers
# =============================================================================
# This will analyze two different implementations: A recursive and iterative one


#%% Recursive
# Similar to for loops, recursive functions are relatively slow, as they call
# themselves potentially a large number of times before coming to a final result.
# In this case, Numba does not help at all, but Cython shows significant speedups 
# based on static type declarations only.

def fib_rec_py1(n):
    if n < 2:
        return n
    else:
        return fib_rec_py1(n - 1) + fib_rec_py1(n - 2)
    
# fib_rec_py1(35) # %time 3.11 s
        
import numba

fib_rec_nb = numba.jit(fib_rec_py1)

#fib_rec_nb(35) # %time 3.59s

#%load_ext Cython
#%%cython
#def fib_rec_cy(int n):
#    if n < 2:
#        return n
#    else:
#        return fib_rec_cy(n-1) + fib_rec_cy(n-2)
#        
#
#%time fib_rec_cy(35) # %time 650 ms

# The major problem with recursive algorithm is that the intermediate results are
# not cached, but rather recalculated.  To avoid this, a decorator can be used
# that takes care of the caching of intermediate results (speeding up by multiple
# orders of magnitude)

from functools import lru_cache as cache

@cache(maxsize = None)
def fib_rec_py2(n):
    if n < 2:
        return n
    else: return fib_rec_py2(n - 1) + fib_rec_py2(n - 2)
    
#fib_rec_py2(35) # %time 0ns
#fib_rec_py2(80) # %time 0ns
    
#%% Iterative
# Here, Numba will help improve, but Cython will be the best

def fib_it_py(n):
    x, y = 0, 1
    for i in range(1, n + 1):
        x, y = y, x + y
    return x

# %time fib_it_py(80) ~ 0ns

import numba
fib_it_nb = numba.jit(fib_it_py)

# %time fib_it_nb(80) 1.22s
# %time fib_it_nb(80) 0 ns


%%cython
def fib_it_cy1(int n):
    cdef long i
    cdef long x = 0, y = 1
    for i in range(1, n + 1):
        x, y = y, x + y
    return x

#%time fib_it_cy1(80) 0ns

#%%time
#fn = fib_rec_py2(150)
#print(fn) -> Output: 9969216677189303386214405760200
#0 ns
#fn.bit_length() # -> 103 which is > 64..Normal python can handle this and provide
#    correct answer
    
#%%time
#fn = fib_it_nb(150)
#print(fn) -> Output: 6792540214324356296 <<< We can see that several numbers missing
# 1 ms
# fn.bit_length() # -> 63

#%%time
#fn = fib_it_cy1(150)
#print(fn) -> 626779336 <<< even less
#0 ns
#fn.bit_length() # -> 30 <<< in book, was actually 63..


#%% The following code does not work...getting system exit error..need to install some other thing?
#%%cython
#cdef extern from *:
#    ctypedef int int128 '__int128_t'
#def fib_it_cy2(int n):
#    cdef int128 i
#    cdef int128 x = 0, y = 1
#    for i in range(1, n + 1):
#        x, y = y, x + y
#    return x

# Above code should be faster and outputting 128-bit int object type that is correct
    