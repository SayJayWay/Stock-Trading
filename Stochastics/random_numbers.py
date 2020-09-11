import math
import numpy as np
import numpy.random as npr
from pylab import plt, mpl

plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

npr.seed(100)
np.set_printoptions(precision = 4)

a = 5.
b = 10.
npr.rand(5,5) * (b - a) + a # Example of making 5 x 5 matrix of random numbers
                            # between 'a' and 'b'

sample_size = 500

rn1 = npr.rand(sample_size, 3) # Uniformly distributed random numbers
rn2 = npr.randint(0, 10, sample_size) # Random integers for given interval
rn3 = npr.sample(size = sample_size)
a = [0, 25, 50, 75, 100]
rn4 = npr.choice(a, size = sample_size) # Random sample values from finite list object

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize = (10,8))

# Visualizing different types of sampling
ax1.hist(rn1, bins = 25, stacked = True)
ax1.set_title('rand')
ax1.set_ylabel('frequency')
ax2.hist(rn2, bins = 25)
ax2.set_title('randint')
ax3.hist(rn3, bins = 25)
ax3.set_title('sample')
ax3.set_ylabel('frequency')
ax4.hist(rn4, bins = 25)
ax4.set_title('choice')

#%%
# Visualizing random draws from following distribution
# Standard normal with mean of 0 and std of 1
# Normal with mean of 100 and standard deviation of 20
# Chi square with 0.5 degrees of freedom
# Poisson with lambda of 1

sample_size = 500
rn1 = npr.standard_normal(sample_size)
rn2 = npr.normal(100, 20, sample_size)
rn3 = npr.chisquare(df = 0.5, size = sample_size)
rn4 = npr.poisson(lam = 1.0, size = sample_size)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize = (10,8))

ax1.hist(rn1, bins = 25)
ax1.set_title('standard normal')
ax1.set_ylabel('frequency')
ax2.hist(rn2, bins = 25)
ax2.set_title('normal(100, 20)')
ax3.hist(rn3, bins = 25)
ax3.set_title('chi square')
ax4.hist(rn4, bins = 25)
ax4.set_title('Poisson')