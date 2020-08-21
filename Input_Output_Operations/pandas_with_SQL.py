# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sqlite3 as sq3

data = np.random.standard_normal((1000000,5)).round(4)

# =============================================================================
# Working with SQL databases
# =============================================================================

filename = './numbers'
con = sq3.Connection(filename + '.db')

query = 'CREATE TABLE numbers (No1 real, No2 real,\
        No3 real, No4 real, No5 real)' # Creates table w/ 5 columns for real numbers

q = con.execute
qm = con.executemany
q(query)

qm('INSERT INTO numbers VALUES (?, ?, ?, ?, ?)', data)
con.commit()

temp = q('SELECT * FROM numbers').fetchall()

query = 'SELECT * FROM numbers WHERE No1>0 and No2 < 0'

res = np.array(q(query).fetchall()).round(3) # ~ 660ms
res = res[::100]

plt.figure(figsize = (10,6))

plt.plot(res[:,0], res[:,1], 'ro')

# =============================================================================
# From SQL to Pandas
# =============================================================================

# Saving all the data from a SQL database using pandas takes same amount of 
# time as using numpy.  It is a lot faster thn when using a SQL disk-based
# approach (out-of-memory).  The bottleneck performance-wise is with the SQL
# database, not pandas

data = pd.read_sql('SELECT * FROM numbers', con) # Save all the data to pandas database

# Now can do all analytics with this dataframe.  Often is a magnitude faster.

data[(data['No1'] > 0) & (data['No2'] < 0)] # ~ 30ms
q = '(No1 < -0.5 | No1 > 0.5) & (No2 < -1 | No2 > 1)'
res = data[['No1', 'No2']].query(q)
plt.figure(figsize = (10,6))
plt.plot(res['No1'], res['No2'], 'ro');
