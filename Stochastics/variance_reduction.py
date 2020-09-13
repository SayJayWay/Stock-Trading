import math
import numpy as np
from pylab import mpl, plt
import numpy.random as npr

print('%15s %15s' % ('Mean', 'Std Deviation'))
print(31 * '-')
for i in range(1, 31, 2):
    npr.seed(100)
    sn = npr.standard_normal(i ** 2 * 10000)
    print('%15.12f %15.12f' % (sn.mean(), sn.std()))

# Can see that (as expected), the statistics 'somehow' get better the larger
# the number of draws get.  There are ways to reduce variance to improve the
# matching of the first two moments of the (standard) normal distribution.
    
#%% Antithetic variates - This approach uses only half the desired number of
# random draws, and adds the same set of random numbers with the opposite sign
# afterward (This method only works for symmetric median 0 random variables only,
# i.e. standard normally distributed random variables -- which are almost exclusively
# used throughout).  For example if the random number generator draws 0.5, then
# another number value -0.5 is added to the set.  By this logic, the mean value of
# of such a dataset must be 0
    
sn = npr.standard_normal(int(10000 / 2))
sn = np.concatenate((sn, -sn))


sn.mean() # 2.842170943040401e-18

print('%15s %15s' % ('Mean', 'Std.Deviation'))
print(31 * "-")
for i in range(1, 31, 2):
    npr.seed(1000)
    sn = npr.standard_normal(i ** 2 * int(10000 / 2))
    sn = np.concatenate((sn, -sn))
    print("%15.12f %12.12f" % (sn.mean(), sn.std()))
    
# Can see from the print function that the mean is 0, but the std is not quite
# 1...Especially as we draw more random numbers.  Can use a variance reduction
# technique called moment matching to correct the first and second moments
    
#%% Moment matching -- Subtracts the mean from every random number and divides
# every single number by the standard deviation.

sn = npr.standard_normal(10000)

sn.mean() # -0.001165998295162494
sn.std() # 0.991255920204605
sn_new = (sn - sn.mean()) / sn.std()
sn_new.mean() # -2.3803181647963357e-17
sn_new.std() # 0.9999999999999999

# Function that generates random numbers for process simulation using either
# none, one or two variance reduction technique(s)
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
        sn = np.concatenate((sn, -sn))
    else:
        sn = npr.standard_normal((M + 1, I))
    if mo_match is True:
        sn = (sn - sn.mean()) / sn.std()
    return sn