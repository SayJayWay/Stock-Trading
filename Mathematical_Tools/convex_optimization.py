import numpy as np
from pylab import plt, mpl

plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

def fm(p):
    x, y = p
    return (np.sin(x) + 0.05 * x ** 2 +
            np.sin(y) + 0.05 * y ** 2)
    
x = np.linspace(-10, 10, 50)
y = np.linspace(-10, 10, 50)
X, Y = np.meshgrid(x, y)
Z = fm((X, Y))

fig = plt.figure(figsize = (10,6))
ax = fig.gca(projection = '3d')
surf = ax.plot_surface(X, Y, Z, rstride = 2, cstride = 2, cmap = 'coolwarm',
                       linewidth = 0.5, antialiased = True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
fig.colorbar(surf, shrink=0.5, aspect=5)

# =============================================================================
# Global Optimization
# Generally will have both a global minimzation approach followed by a local one
# =============================================================================
# Will use sco.brute() and sco.fmin() from the scipy.optimize library

import scipy.optimize as sco

def fo(p):
    x, y = p
    z = np.sin(x) + 0.05 * x ** 2 + np.sin(y) + 0.05 * y ** 2
    if output == True:
        print('%8.4f | %8.4f | %8.4f' % (x, y, z))
    return z

output = True
sco.brute(fo, ((-10, 10.1, 5), (-10, 10.1, 5)), finish = None) # array([0., 0.])

# From this array, can see it is near x = y = 0, however, we used step size of 5, 
# which is quite large...let's try 0.1
output = False
opt1 = sco.brute(fo, ((-10, 10.1, 0.1), (-10, 10.1, 0.1)), finish = None) # array([-1.4, -1.4])
z_value = fm(opt1)

# Can calculate local minimum around - x = y -1.4
output = True
opt2 = sco.fmin(fo, (-1.4, -1.4), maxiter = 250)

# =============================================================================
# Constrained Optimization
# Previous example were unconstrained.  In economics/financial problems, there
# are one or multiple constraints which can form of equalities or inequalities
# =============================================================================

#%% Example
#Investor who can invest in two risky securities.  Both securities cost q_a = q_b = 10USD
#After one year, they have a payoff of 15USD and 5USD in state u and 5USD and 12USD,
#respectively in state d.  Both states are equally likely.  Denote the vector payoffs
#for the two securities by r_a and r_b, respectively
#
#Investor has budget of w_0 = 100 USD and derives utility from future wealth according
#to the utility function u(w) = sqrt(w) where w is the wealth available.
#Use expected utility maximization formula (pg 333)

import math

def Eu(p): # Function to be minimized in order to maximize expected utility
    s, b = p
    return -(0.5 * math.sqrt(s * 15 + b * 5) +
             0.5 * math.sqrt(s * 5 + b * 12))
    
cons = ({'type' : 'ineq', # inequality constraint as a dictionary object
         'fun' : lambda p: 100 - p[0] * 10 - p[1] * 10})
    
bnds = ((0, 1000), (0, 1000)) # Boundaries - chosen to be wide enough

result = sco.minimize(Eu, [5, 5], method='SLSQP',
                      bounds=bnds, constraints=cons)


#     fun: -9.700883611487832 # negative min function value as optimal solution value
#     jac: array([-0.48508096, -0.48489535])
# message: 'Optimization terminated successfully.'
#    nfev: 21
#     nit: 5
#    njev: 5
#  status: 0
# success: True
#       x: array([8.02547122, 1.97452878]) # Optimal parameter values (i.e. optimal portfolio values)

np.dot(result['x'], [10, 10]) # 99.99999999 -> Budget constraint is binding; all wealth is invested
