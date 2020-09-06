# =============================================================================
# Pi
# =============================================================================

# This algorithm will be a Monte Carlo siulation-based algorithm to derive the
# digits for pi.  The basic idea is that the area A of a circle is given by 
# A = pi*r^2 and therefore pi = A/r^2.  For a unit circly, pi = A.

# The idea of the alrgorithm is to simulate random points with coordinate
# values (x,y) where x,y E [-1,1].  The area of an origin-centered square with
# side length of 2 is exactly 4.  The area of the origin-centered unit circle is
# a fraction of the area of such a square.  This fraction can be estimated by
# Monte Carlo simulation: count all the points in the square, then count all the
# points in the circle, and divide the number of points in the circle by the 
# number of points in the square

import random
import numpy as np
from pylab import mpl, plt

plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'
#%matplotlib inline

rn = [(random.random() * 2 - 1, random.random() * 2 - 1) for _ in range(500)]
rn = np.array(rn) # Random dots between -1 and 1

# Draw figures
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(1, 1, 1)
circ = plt.Circle((0 ,0), radius = 1, edgecolor = 'g', lw=2.0, facecolor = 'None')
box = plt.Rectangle((-1, -1), 2, 2, edgecolor = 'b', alpha = 0.3)
ax.add_patch(circ)
ax.add_patch(box)
plt.plot(rn[:, 0], rn[:, 1], 'r.')
plt.ylim(-1.1, 1.1)
plt.xlim(-1.1, 1.1)

#%% Numpy implementation is concise but also memory intensive
n = int(1e7)

rn = np.random.random((n, 2)) * 2 - 1 # %time 379ms

# Distance of the points from the origin (Euclidean norm)
distance = np.sqrt((rn ** 2).sum(axis=1)) # %time 336ms

# Calculating fraction of those points on the circle relative to all points
frac = (distance <= 1.0).sum() / len(distance) # %time 20.2ms

pi_mcs = frac * 4 # This accounts for the square area of 4 for the estimation
                  # of the circle area and therewith of pi
print('pi_mcs = {}'.format(pi_mcs))

#%% pi_mcs is a Python function using a for loop and implementing the Monte Carlo
#   simulation in a memory-efficient manner.  Note that the random numbers are 
#   not scaled here.  Execution time is > than w/ Numpy, but Numba version is faster
#   than numpy in this case

# Regular for loop
def mcs_pi_py(n):
    circle = 0
    for _ in range(n):
        x, y = random.random(), random.random()
        if (x ** 2 + y ** 2) ** 0.5 <= 1:
            circle += 1
    return (4 * circle) / n
    
x = mcs_pi_py(n) # %time 7.32s
print('x = {}'.format(x))

#%% With numba

import numba

mcs_pi_nb = numba.jit(mcs_pi_py)

mcs_pi_nb #%time 659ms
mcs_pi_nb #%time 124ms

#%% A plain Cython w/ stati declations is not much faster than Py, but relying 
#   on random number generation capabilities of C further speeds up the calculations


# =============================================================================
# STATIC DECLARATION
# %load_ext cython
# %%cython -a
# import random
# def mcs_pi_cy1(int n):
#     cdef int i, circle = 0
#     cdef float x, y
#     for i in range(n):
#         x, y = random.random(), random.random()
#         if (x ** 2 + y ** 2) ** 0.5 <= 1:
#             circle += 1
#     return (4 * circle) / n
# 
# mcs_pi_cy1(n) #%time 1.37s
# =============================================================================


# =============================================================================
# Don't understand how this is supposed to work...INT_MAX gives max possible int..2147483647
# %%cython -a
# from libc.stdlib cimport rand
# cdef extern from 'limits.h':
#     int INT_MAX
#     
# def mcs_pi_cy2(int n):
#     cdef int i, circle = 0
#     cdef float x, y
#     for i in range(n):
#         x, y = rand() / INT_MAX, rand() / INT_MAX
#         if (x ** 2 + y ** 2) ** 0.5 <= 1:
#             circle += 1
#     return (4 * circle) / n
# =============================================================================

