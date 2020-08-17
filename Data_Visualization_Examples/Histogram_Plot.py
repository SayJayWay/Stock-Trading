import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1000)
y = np.random.standard_normal((1000,2))

plt.figure(figsize = (10,6))
plt.hist(y, label=['1st', '2nd'] ,bins=25)
plt.legend(loc=0)
plt.xlabel('value')
plt.ylabel('frequency')
plt.title('Simple Histogram Plot')
plt.show()

# Stacked Histogram
plt.figure(figsize=(10,6))
plt.hist(y, label=['1st', '2nd'], color=['b', 'g'],
            stacked = True, bins=20, alpha=0.5)
plt.legend(loc=0)
plt.xlabel('value')
plt.ylabel('frequency')
plt.title('Stacked Histogram Plot')
plt.show()
