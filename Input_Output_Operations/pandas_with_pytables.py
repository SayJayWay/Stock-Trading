# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tables as tb # Pytables import name is 'tables'
import datetime as dt

# Pytables is Python binding for HDF5 database standard.

filename = './pytab.h5'

h5 = tb.open_file(filename, 'w')

row_des = {
        'Date' : tb.StringCol(26, pos = 1),
        'No1' : tb.IntCol(pos = 2),
        'No2' : tb.IntCol(pos = 3),
        'No3' : tb.Float64Col(pos = 4),
        'No4' : tb.Float64Col(pos = 5)
        }

rows = 2000000
filters = tb.Filters(complevel = 0) # compression level = 0

tab = h5.create_table('/', 'ints_floats', # The node (path) and technical name of table
                      row_des, # Description of row data structure
                      title = 'Integers and Floats', # Name of table
                      expectedrows = rows, # expected # of rows (allows for optimization)
                      filters = filters)

pointer = tab.row # Pointer object

ran_int = np.random.randint(0, 10000, size=(rows,2)) 

ran_flo = np.random.standard_normal((rows, 2)).round(4)

# datetime object and two int/two float objects are written row-by-row and appended
# This is a slow way to append rows
for i in range(rows):
    pointer['Date'] = dt.datetime.now()
    pointer['No1'] = ran_int[i, 0]
    pointer['No2'] = ran_int[i, 1]
    pointer['No3'] = ran_flo[i, 0]
    pointer['No4'] = ran_flo[i, 1]
    pointer.append()
    
tab.flush() # All rows are flushed (committed as permanent changes)

# The more performant/Pythonic way is by using NumPy structured arrays

dty = np.dtype([('Date', 'S26'), ('No1', '<i4'), ('No2', '<i4'),
                ('No3', '<f8'), ('No4', '<f8')])
    
sarray = np.zeros(len(ran_int), dtype = dty)

sarray['Date'] = dt.datetime.now()
sarray['No1'] = ran_int[:,0]
sarray['No2'] = ran_int[:,1]
sarray['No3'] = ran_flo[:,0]
sarray['No4'] = ran_flo[:,1]

# Create table and populate it with data
h5.create_table('/', 'ints_float_from_array', sarray,
                title = 'Integers and Floats',
                expectedrows = rows,
                filters = filters)

# h5.remove_node('/', 'ints_floats_from_array') # Removes second Table w/ the redundant data

# Table object behaves similar to NumPy ndarray in most cases as shown below:
tab[:3]
tab[:4]['No4']
np.sum(tab[:]['No3'])
np.sum(np.sqrt(tab[:]['No1']))
plt.figure(figsize=(10,6))
plt.hist(tab[:]['No3'], bins = 30);

# =============================================================================
# Using PyTables with SQL-like statements
# =============================================================================

query = '((No3 < -0.5) | (No3 > 0.5)) & ((No4 < -1) | (No4 > 1))'

iterator = tab.where(query) # Creating iterator object based on query

res = [(row['No3'], row['No4']) for row in iterator] 
res = np.array(res)
res[:3]
plt.figure(figsize = (10,6))
plt.plot(res.T[0], res.T[1], 'ro')

# =============================================================================
# Working with PyTables as Table objects is similar to Numpy/pandas objects in
# terms of both syntax and performance
# =============================================================================

values = tab[:]['No3']
print('Max %18.3f' % values.max())
print('Ave %18.3f' % values.mean())
print('Min %18.3f' % values.min())
print('Std %18.3f' % values.std())

res = [(row['No1'], row['No2']) for row in
       tab.where('((No1 > 9800) | (No1 < 200)) \
                 & ((No2 > 4500) & (No2 < 500))')]

for r in res[:4]:
    print(r)
    
res = [(row['No1'], row['No2']) for row in
       tab.where('(No1 == 1234) & (No2 > 9776)')]

for r in res:
    print(r)