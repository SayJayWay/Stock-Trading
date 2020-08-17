import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1000)

def func(x):
    return 0.5 * np.exp(x) + 1
a, b = 0.5, 1.5
x = np.linspace(0,2)
y = func(x)
Ix = np.linspace(a,b)
Iy = func(Ix)
verts = [(a,0)] + list(zip(Ix, Iy)) + [(b,0)]

from matplotlib.patches import Polygon
fig, ax = plt.subplots(figsize=(10,6))
plt.plot(x, y, 'b', linewidth=2)
plt.ylim(bottom = 0)
poly = Polygon(verts, facecolor = '0.7', edgecolor = '0.5')
ax.add_patch(poly)
plt.text(0.5 * (a + b), 1, r'$\int_a^b f(x)\mathrm{d}x$',
        horizontalalignment='center', fontsize = 20)
plt.figtext(0.9, 0.075, '$x$')
plt.figtext(0.075, 0.9, '$f(x)$')
ax.set_xticks((a, b))
ax.set_xticklabels(('$a$', '$b$'))
ax.set_yticks([func(a), func(b)])
ax.set_yticklabels(('$f(a)$', '$f(b)$'))
plt.show()
