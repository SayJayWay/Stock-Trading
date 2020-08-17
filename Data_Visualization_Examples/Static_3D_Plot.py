import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Prepping the data
strike = np.linspace(50, 150, 24)
ttm = np.linspace(0.5 ,2.5, 24)
strike, ttm = np.meshgrid(strike, ttm)
iv = (strike - 100) ** 2 / (100 * strike) / ttm

# Plotting
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize = (10,6))
ax = fig.gca(projection = '3d')
surf = ax.plot_surface(strike, ttm, iv, rstride = 2, cstride = 2,
                        cmap=plt.cm.coolwarm, linewidth = 0.5, antialiased = True)
ax.set_xlabel('strike')
ax.set_ylabel('time-to-maturity')
ax.set_zlabel('implied volatility')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

# Can make it scatter 3D plot too
fig = plt.figure(figsize = (10,6))
ax = fig.add_subplot(111, projection = '3d')
ax.view_init(30, 60)
ax.scatter(strike, ttm, iv, zdir='z', s=25,
            c='b', marker = '^')
ax.set_xlabel('strike')
ax.set_ylabel('time-to-maturity')
ax.set_zlabel('implied volatility')

plt.show()
