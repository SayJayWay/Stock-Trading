# Two ways to handle data of different magnitudes
# 1) Use two y-axes (left/right)
# 2) Use two subplots (upper/lower, left/right)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1000)
y = np.random.standard_normal((20,2))

# With subplorts
def withTwoYAxis():
    fig, ax1 = plt.subplots() # Defines figure and axis objects
    plt.plot(y[:,0], 'b', lw=1.5, label='1st')
    plt.plot(y[:,0], 'ro')
    plt.legend(loc=8)
    plt.xlabel('index')
    plt.ylabel('value')
    plt.title('Using Two Y-Axes')
    ax2 = ax1.twinx() # Creates secnd axis object that shares the x-axis
    plt.plot(y[:,1], 'g', lw=1.5, label='2nd')
    plt.plot(y[:,1], 'ro')
    plt.legend(loc=0)
    plt.ylabel('value 2nd');
    plt.show()

def withSubPlots():
    fig, ax1 = plt.subplots()
    plt.subplot(211) # numrows, numcols, fignum
    plt.plot(y[:,0], lw=1.5, label='1st')
    plt.plot(y[:,0], 'ro')
    plt.legend(loc=0)
    plt.ylabel('value')
    plt.title('Using Sub Plots')
    plt.subplot(212)
    plt.plot(y[:,1], 'g', lw=1.5, label='2nd')
    plt.plot(y[:,1], 'ro')
    plt.legend(loc=0)
    plt.xlabel('index')
    plt.ylabel('value');
    plt.show()

#withTwoYAxis()
withSubPlots()
