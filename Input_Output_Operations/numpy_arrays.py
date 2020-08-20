# -*- coding: utf-8 -*-

# =============================================================================
# With standard speed of SSD, numpy can write ~500MB/s (much faster than pickle
# module for serialization.  There are two reasons for this:
#                1) The data is numeric
#                2) NumPy uses binary storage
# Binary storage will reduce the overhead almost to zero.  The only flaw from
# this approach would be the lack of functionality that a SQL database would
# have.  PyTables will help in this regards.
# =============================================================================

import numpy as np

dtimes = np.arange('2019-01-01 10:00:00', '2025-12-31 22:00:00', dtype = 'datetime64[m]')
dty = np.dtype([('Date', 'datetime64[m]'),
                ('No1', 'f'), ('No2', 'f')])

data = np.zeros(len(dtimes), dtype=dty)
data['Date'] = dtimes

a = np.random.standard_normal((len(dtimes), 2)).round(4)
data['No1'] = a[:,0]
data['No2'] = a[:,1]

np.save('./array', data) # Saves data
np.load('./array.npy')
