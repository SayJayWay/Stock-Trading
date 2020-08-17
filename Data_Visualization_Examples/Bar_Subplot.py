import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1000)
y = np.random.standard_normal((20,2))

plt.figure(figsize=(10,6))
plt.subplot(121)
plt.plot(y[:,0], lw = 1.5, label = '1st')
plt.plot(y[:,0], 'ro')
plt.legend(loc=0)
plt.xlabel('index')
plt.ylabel('value')
plt.title('1st Data Set')

plt.subplot(122)
plt.bar(np.arange(len(y)), y[:,1], width = 0.5, color = 'g', label = '2nd')
plt.legend(loc = 0)
plt.xlabel('index')
plt.title('2nd Data Set')

plt.show()
