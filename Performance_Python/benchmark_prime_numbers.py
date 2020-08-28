# =============================================================================
# The following algorithms are regularly used for performance benchmarks
# =============================================================================

# =============================================================================
# Prime Numbers
# =============================================================================

## This is a hard coded way to find prime numbers

def is_prime(I):
    if I % 2 == 0: return False
    for i in range(3, int(I**0.5) + 1, 2):
        if I % i == 0: return False
    return True

n = int(1e8+3)

#is_prime(n) # %time 0 ns

p1 = int(1e8+7)

#is_prime(n) # %time 0 ns

p2 = 10010910012962907

#is_prime(p2) # %time 0 ns (in the book it took 22.7s...)

## Using Numba
import numba

is_prime_nb = numba.jit(is_prime)

#is_prime_nb(n) # %time 395 ms
#is_prime_nb(n) # %time 0 ns

#is_prime_nb(p1) # %time 0 ns

#is_prime_nb(p2) # %time 0 ns -> should be order of magnitude faster, but my computer
                 # already solves is fairly quickly
                 
#%% Cython
#
#%load_ext Cython
#%%cython
#def is_prime_cy1(I):
#    if I % 2 == 0: return False
#    for i in range(3, int(I**0.5) + 1, 2):
#        if I % i == 0: return False
#    return True
#
#%timeit is_prime(p1) # 329 µs ± 51.5 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
#
#%timeit is_prime_cy1(p1) # 164 µs ± 1.54 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
                 
# Using cpython is already faster than the original when using the same code
                 
                 
## Can be faster if we use static type declaration.  Doing this can be faster than Numba

%%cython
def is_prime_cy2(long I): # static type declaration for I
    cdef long i # static type declaration for i
    if I % 2 == 0: return False
    for i in range(3, int(I ** 0.5) + 1, 2):
        if I % i == 0: return False
    return True

#%timeit is_prime_cy2(p1) # 12.5 µs ± 57.3 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    
# %time is_prime_cy2(p2) # I get an error: OverflowError: Python int too large to convert to C long
    


#%% Multiprocessing
# First an mp.Pool object is set up with multiple processes.  Second, the function
# to be executed is mapped to the prime numbers to be checked

import multiprocessing as mp

if __name__ == '__main__':
    pool = mp.Pool(processes=4)
    pool.map(is_prime, 10 * [p1]) # %time 4.48ms
    
    pool.map(is_prime_nb, 10 * [p2]) # %time 325ms
        
    pool.map(is_prime_cy2, 10 * [p2]) # %time -> this one hangs...
    
# Observed speedup should be quicker.  Parallel processing should be considered whenever
# different problems of the same type need to be solved.  The effect can be substantial
# when many cores and memory is available.  multiprocessing modul is also easy to use
    
    



