import matplotlib as matplotlib
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1000)
y = np.random.standard_normal((20,2)).cumsum(axis=0)

plt.figure(figsize=(10,60))

# If they have the same x-axis we can do it without any 'x' values
plt.plot(y[:,0], lw = 1.5, label='1st')
plt.plot(y[:,1], lw = 1.5, label='2nd')
plt.plot(y, 'ro')
plt.xlabel('index')
plt.ylabel('value')
plt.title('A Simple Plot')
# Loc = 0 means legend will block the least amount of data
plt.legend(loc=0)
plt.show()
