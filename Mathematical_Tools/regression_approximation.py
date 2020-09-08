import numpy as np
from pylab import plt, mpl

plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

# Standard function we will use to test
def f(x):
    return np.sin(x) + 0.5*x

# Standard function to create plot
def create_plot(x, y, styles, labels, axlabels):
    plt.figure(figsize=(10,6))
    for i in range(len(x)):
        plt.plot(x[i], y[i], styles[i], label=labels[i])
        plt.xlabel(axlabels[0])
        plt.ylabel(axlabels[1])
    plt.legend(loc=0)
    
x = np.linspace(-2 * np.pi, 2 * np.pi, 50)

create_plot([x], [f(x)], ['b'], ['f(x)'], ['x', 'f(x)'])

# Simple Linear Regression (on polynomial)
res = np.polyfit(x, f(x), deg=1, full=True)
ry = np.polyval(res[0], x)
create_plot([x, x], [f(x), ry], ['b', 'r.'], ['f(x)', 'regression'], ['x', 'f(x)'])

# Simple Regression (degree = 5)
reg = np.polyfit(x, f(x), deg=5)
ry = np.polyval(reg, x)
create_plot([x, x], [f(x), ry], ['b', 'r.'], ['f(x)', 'regression degree 5'], ['x', 'f(x)'])

# Simple Regression (degree = 7)
reg = np.polyfit(x, f(x), 7)
ry = np.polyval(reg, x)
create_plot([x, x], [f(x), ry], ['b', 'r.'], ['f(x)', 'regression degree 7'], ['x', 'f(x)'])

# Test if regression is close to function
close_test = np.allclose(f(x), ry) # -> Will return True if close

MSE = np.mean((f(x) - ry) ** 2) # -> Mean Squared Error

#%%
# =============================================================================
# Individual Basis Functions
# Can exploit our knowledge about the function to approximate
# =============================================================================

#%% Standard base function approximating with x^3, x^2, x, etc
matrix = np.zeros((3 + 1, len(x))) # ndarray object for basis function values (matrix)

# Basis function values from constant to cubic
matrix[3, :] = x ** 3
matrix[2, :] = x ** 2
matrix[1, :] = x
matrix[0, :] = 1

reg = np.linalg.lstsq(matrix.T, f(x), rcond=None)[0] # Regression parameters

ry = np.dot(reg, matrix) # Regression estimates for function values

create_plot([x, x], [f(x), ry], ['b', 'r.'], ['f(x)', 'Regression basis function'], ['x', 'f(x)'])

#%% Since we know the function has a sin, let's include a sin term

matrix[3, :] = np.sin(x)

reg = np.linalg.lstsq(matrix.T, f(x), rcond=None)[0]
ry = np.dot(reg, matrix)

create_plot([x, x], [f(x), ry], ['b', 'r.'], ['f(x)', 'Regression basis function w/ sin'], ['x', 'f(x)'])

#%% 
# =============================================================================
# Dealing with Noisy Data
# Numpy regression can deal w/ noisy data 'decently'
# =============================================================================

xn = np.linspace(-2 * np.pi, 2 * np.pi, 50)
xn = xn + 0.15 * np.random.standard_normal(len(xn))
yn = f(xn) + 0.25 * np.random.standard_normal(len(xn))

reg = np.polyfit(xn, yn, 7)
ry = np.polyval(reg, xn)

create_plot([x, x], [f(x), ry], ['b', 'r.'], ['f(x)', 'Regression Noisy Data'], ['x', 'f(x)'])

#%%
# =============================================================================
# Unsorted Data
# =============================================================================
xu = np.random.rand(50) * 4 * np.pi - 2 * np.pi
yu = f(xu)

reg = np.polyfit(xu, yu, 5)
ry = np.polyval(reg, xu)

create_plot([xu, xu], [yu, ry], ['b.', 'ro'], ['f(x)', 'Regression unsorted Data'], ['x', 'f(x)'])

#%% 
# =============================================================================
# Multiple dimensions
# Numpy (least-squares regression approach) can handle multiple dimensions quite well
# =============================================================================

def fm(p):
    x, y = p
    return np.sin(x) + 0.25 * x + np.sqrt(y) + 0.05 * y ** 2

x = np.linspace(0, 10, 20)
y = np.linspace(0, 10, 20)

X, Y = np.meshgrid(x, y) # Generates 2D ndarray objects out of the 1D ndarray objects

Z = fm((X, Y))
x = X.flatten() # Flattens list of list into a single list
y = Y.flatten()

# Visualization the function fm(p)
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize = (10,6))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride = 2, cstride = 2, cmap = 'coolwarm',
                       linewidth = 0.5, antialiased = True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
fig.colorbar(surf, shrink = 0.5, aspect = 5)

# Performing regression on fm(p)
matrix = np.zeros((len(x), 6 + 1))
matrix[:, 6] = np.sqrt(y)
matrix[:, 5] = np.sin(x)
matrix[:, 4] = y ** 2
matrix[:, 3] = x ** 2
matrix[:, 2] = y
matrix[:, 1] = x
matrix[:, 0] = 1

reg = np.linalg.lstsq(matrix, fm((x, y)), rcond = None)[0]

RZ = np.dot(matrix, reg).reshape((20, 20))

fig = plt.figure(figsize=(10, 6))
ax = fig.gca(projection = '3d')
surf1 = ax.plot_surface(X, Y, Z, rstride = 2, cstride = 2, cmap = mpl.cm.coolwarm,
                        linewidth = 0.5, antialiased = True)
surf2 = ax.plot_wireframe(X, Y, RZ, rstride = 2, cstride = 2, label = 'regression')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.legend()
fig.colorbar(surf, shrink=0.5, aspect=5)

# =============================================================================
# Interpolation (e.g. with cubic splines)
# Continuous differentiability requires at least interpolation of degree 3
# =============================================================================

import scipy.interpolate as spi
x = np.linspace(-2 * np.pi, 2 * np.pi, 25)

def f(x):
    return np.sin(x) + 0.5 * x

ipo = spi.splrep(x, f(x), k = 1)
iy = spi.splev(x, ipo)

print(np.allclose(f(x), iy))

create_plot( [x, x], [f(x), iy], ['b', 'ro'], ['f(x)', 'interpolation'], ['x', 'f(x)'])

# Spline interpolation often used in finance to generate estimates for dependent 
# values of independent data points not included in the original observations.
# Next example picks smaller interval and has closer look at interpolated values
# with the linear splines

xd = np.linspace(1.0, 3.0, 50)
iyd = spi.splev(xd, ipo)

create_plot([xd, xd], [f(xd), iyd], ['b', 'ro'], ['f(x)', 'interpolation'],
            ['x', 'f(x)'])

ipo = spi.splrep(x, f(x), k=3)
iyd = spi.splev(xd, ipo)

np.allclose(f(xd), iyd) # False
np.mean((f(xd) - iyd) **2)

create_plot([xd, xd], [f(xd), iyd], ['b', 'ro'], ['f(x)', 'interpolation'],
            ['x', 'f(x)'])