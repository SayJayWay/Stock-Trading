# -*- coding: utf-8 -*-

from pylab import plt, mpl
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

import pickle
import numpy as np
from random import gauss

a = [gauss(1.5, 2) for i in range (1000000)]
pkl_file = open('data.pkl', 'wb')
pickle.dump(np.array(a), pkl_file)
pickle.dump(np.array(a) ** 2, pkl_file)

pkl_file.close()

pkl_file = open('data.pkl', 'rb')

# Will load them on first to enter is first to leave basis
x = pickle.load(pkl_file)
y = pickle.load(pkl_file)
pkl_file.close()

# =============================================================================
# Note that instead of storing multiple objects, it is good practice to instead
# load a dictionary and only have one thing to unload
# =============================================================================
