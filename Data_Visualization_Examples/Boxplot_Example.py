import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1000)
y = np.random.standard_normal((1000,2))

fig, ax = plt.subplots(figsize = (10,6))
plt.boxplot(y)
plt.setp(ax, xticklabels = ['1st', '2nd']) # Can set various parameters using plt.setp instead
plt.xlabel('data set')
plt.ylabel('value')
plt.title('Boxplot')
plt.show()
