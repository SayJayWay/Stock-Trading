# Integration 

import scipy.integrate as sci
from pylab import plt, mpl
import numpy as np

def f(x):
    return np.sin(x) + 0.5 * x

x = np.linspace(0, 10)
y = f(x)
a = 0.5
b = 9.5
Ix = np.linspace(a, b)
Iy = f(x)

from matplotlib.patches import Polygon

fig, ax = plt.subplots(figsize = (10,6))
plt.plot(x, y, 'b', linewidth = 2)
plt.ylim(bottom = 0)
Ix = np.linspace(a,b)
Iy = f(Ix)

verts = [(a,0)] + list(zip(Ix, Iy)) + [(b, 0)]
poly = Polygon(verts, facecolor = '0.7', edgecolor='0.5')
ax.add_patch(poly)
plt.text(0.75 * (a + b), 1.5, r"$\int_a^b f(x)dx$",
         horizontalalignment='center', fontsize=20)
plt.figtext(0.9, 0.075, '$x$')
plt.figtext(0.075, 0.9, '$f(x)$')
ax.set_xticks((a, b))
ax.set_xticklabels(('$a$', '$b$'))
ax.set_yticks([f(a), f(b)]);

#%%
# =============================================================================
# Numerical Integration
# scipy.integrate subpackage contains several functions to numerically integrate
# for fixed Gaussian quadrature - sci.fixed_quad(),
# for adaptive quadrature - sci.quad(),
# for Romberg integration - sci.romberg()
# =============================================================================

sci.fixed_quad(f, a, b)[0]
sci.quad(f, a, b)[0]
sci.romberg(f, a, b)

# there are also functions that take as input list or ndarray objects w/
# function values and input values, respectively.  example include:
# trapezoidal rule - sci.trapz()
# Simpson's rule - sci.simps()

xi = np.linspace(0.5, 9.5, 25)
sci.trapz(f(xi), xi)
sci.simps(f(xi), xi)

#%%
# =============================================================================
# Simulation Integration
# To evaluate integration ufnction by simulation, draw I random values of x b/w
# integral limits and evaluate function at every random value for x.  Sum up all
# function values and take average to arrive at an average function value of the
# integration interval.  Multiply this value by the length of the integration
# interval to derive an estimate for the integral value.
# =============================================================================

for i in range(1,20):
    np.random.seed(1000)
    # Number of random x values increases with each iteration
    x = np.random.random(i * 10) * (b-a) + a
    print(np.mean(f(x)) * (b-a))

#%%
# =============================================================================
# Symbolic Computation
# =============================================================================
import sympy as sy

# Sympy automatically simplifies given mathematical expressions

x = sy.Symbol('x')
y = sy.Symbol('y')
sy.sqrt(x)
3 + sy.sqrt(x) - 4 ** 2 # Output: sqrt(x) - 13
f = x ** 2 + 3 + 0.5 * x ** 2 + 3/2

sy.simplify(f) # Output: 1.5 * x ** 2 + 4.5

# Can also do pretty printing
sy.init_printing(pretty_print = False, use_unicode = False)
print(sy.pretty(f))

print(sy.pretty(sy.sqrt(x) + 0.5))

# SymPy easily provide large numbers of pi
pi_str = str(sy.N(sy.pi, 400000)) # %time 1.26s
pi_str.find('061072') # finds string '061072' in string -> %time 0ns

# Solving equations
sy.solve(x ** 2 - 1)
sy.solve(x ** 2 - 1 -3)
sy.solve(x ** 3 + 0.5 * x ** 2 - 1)
sy.solve(x ** 2 + y ** 2)

#%% Using Sympy for Integration and Differentiation
a, b = sy.symbols('a b')
I = sy.Integral(sy.sin(x) + 0.5 * x, (x, a, b))

print(sy.pretty(I))

int_func = sy.integrate(sy.sin(x) + 0.5 * x, x)

print(sy.pretty(int_func))

Fb = int_func.subs(x, 9.5).evalf()
Fa = int_func.subs(x, 0.5).evalf()

output = Fb - Fa # 24.3747547180867

# Can also be solved symbolically with symbolic integration limits
int_func_limits = sy.integrate(sy.sin(x) + 0.5 * x, (x, a, b))
int_func_limits.subs({a: 0.5, b : 9.5}).evalf() # 24.3747547180868

sy.integrate(sy.sin(x) + 0.5 * x, (x, 0.5, 9.5)) # 24.3747547180867

# Differentiation
int_func.diff()

# Since a necessary condition for a minimum is that both partial derivatives
# are zero, we can use this to solve for it

f = (sy.sin(x) + 0.05 * x ** 2 +
     sy.sin(y) + 0.05 * y ** 2)

del_x = sy.diff(f, x)
print(del_x)
del_y = sy.diff(f, y)

xo = sy.nsolve(del_x, -1.5) # educated guess for root
yo = sy.nsolve(del_y, -1.5) # educated guess for root

print(xo, yo) # -1.42755177876459 -1.42755177876459

out = f.subs({x : xo, y : yo}).evalf()

print(out) # -1.77572565314742

# Based on educated guess, can get stuck in a local minimum instead of global one

xo = sy.nsolve(del_x, 1.5)
yo = sy.nsolve(del_y, 1.5)
out2 = f.subs({x : xo, y : yo}).evalf()
print(out2) # 2.27423381055640