import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1000)
y = np.random.standard_normal((1000,2))

plt.figure(figsize = (10,6))
plt.scatter(y[:,0], y[:,1], marker = 'o')
plt.xlabel('1st')
plt.ylabel('2nd')
plt.title('Simple Scatter Plot')
plt.show()

# Adding a 3rd dimension
c = np.random.randint(0, 10, len(y))
plt.figure(figsize=(10,6))
plt.scatter(y[:,0], y[:,1],
            c=c,
            cmap = 'coolwarm',
            marker = 'o')
plt.colorbar()
plt.xlabel('1st')
plt.ylabel('2nd')
plt.title('Scatter Plot')
plt.show()
