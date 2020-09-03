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