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