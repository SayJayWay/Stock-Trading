import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

np.random.seed(100)

y = np.random.standard_normal(20)
x = np.arange(len(y))
plt.plot(y)

# Optional Parameters
plt.grid(False)
plt.axis('equal')
plt.plot(y.cumsum())
plt.xlim(-1, 20)
plt.ylim(np.min(y.cumsum()) - 1, np.max(y.cumsum()) + 1);

# Alternate plotting
plt.figure(figsize=(10,6))
plt.plot(y.cumsum(), 'b', lw=1.5)
plt.plot(y.cumsum(), 'ro')
plt.xlabel('index')
plt.ylabel('vvalue')
plt.title('A Simple Plot');

plt.show()
