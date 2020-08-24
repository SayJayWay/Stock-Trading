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
       tab.where('(No1 == 1234) & (No2 > 9776)')] # %time this

for r in res:
    print(r)
    

# =============================================================================
# Won't see much change when using a standard SSD in terms of performance of I/O
# but will help in certain hardware scenarios.  In general there is almost no 
# "disadvantage" to using compression.  Below we can see performance on a
# compressed table
# =============================================================================

filename = './pytabc.h5'

h5c = tb.open_file(filename, 'w')

filters = tb.Filters(complevel = 5, # compression level b/w 0 (low) and 9 (high))
                     complib = 'blosc') # http://blosc.org for more info

rows = 2000000
ran_int = np.random.randint(0, 10000, size=(rows,2)) 
ran_flo = np.random.standard_normal((rows, 2)).round(4)

dty = np.dtype([('Date', 'S26'), ('No1', '<i4'), ('No2', '<i4'),
                ('No3', '<f8'), ('No4', '<f8')])
    
sarray = np.zeros(len(ran_int), dtype = dty)

sarray['Date'] = dt.datetime.now()
sarray['No1'] = ran_int[:,0]
sarray['No2'] = ran_int[:,1]
sarray['No3'] = ran_flo[:,0]
sarray['No4'] = ran_flo[:,1]

tabc = h5c.create_table('/', 'ints_floats', sarray,
                        title = 'Integers and Floats',
                        expectedrows = rows,
                        filter = filters)

query = '((No3 < -0.5) | (No3 > 0.5)) & ((No4 < -1) | No4 > 1))'

iteratorc = tabc.where(query)

res = [(row['No3'], row['No4']) for row in iteratorc] # %time this
# Can see that doing analytics on a compressed table is ~slightly slower
# compared to the uncompressed Table object

res = np.array(res)
res[:3]

# =============================================================================
# Now if we compare with an ndarray object (below)
# =============================================================================

arr_non = tab.read()

tab.size_on_disk
arr_non.nbytes

arr_com = tabc.read() # %time this

tabc.size_on_disk
arr_com.nbytes

h5c.close()

# Can see hardly any speed difference b/w compressed and uncompressed Tables,
# however, file sizes can be significantly reduced when using compressed
# Benefits:
# - Storage costs reduced
# - Backup costs reduced
# - Network traffic reduced
# - Network speed is improved (i.e. storage on and retrieval from servers)
# - CPU utilization increased to overcome I/O bottlenecks

# =============================================================================
# PyTables is also fast/efficient when storing/retrieving ndarray objects
# =============================================================================

# %%time
arr_int = h5.create_array('/', 'integers', ran_int)
arr_flo = h5.create_array('/', 'floats', ran_flo)

# Writing those two objects directly to an HDF5 db is faster than looping over
# the objects and writing the data row-by-row or using the approach via structured
# ndarray objects

# =============================================================================
# PyTables can support out-of-memory operations (array-based computations that
# do not fit in memory.)
# =============================================================================

filename = 'erray.h5'
h5 = tb.open_file(filename, 'w')
n = 500 #num of columns
ear = h5.create_array('/', 'ear',
                      atom = tb.Float64Atom(), # Atomic dtype object of the single values
                      shape = (0,n)) # shape for instantiation (no rows, n columns)

type(ear)

rand = np.random.standard_normal((n,n))

# %%time
for _ in range(750):
    ear.append(rand)

ear.flush()
ear

ear.size_on_disk

# For out-of-memory computations that do not lead to aggregations, another
# EArray object of the same shape (size) is needed.  PyTables can cope w/ 
# numerical expressions using a special module called Expr (based on numerical
# expression library numexpr - https://numexpr.readthedocs.io)

# Example calculation on the ear variable with the expression:
# y = 3*sin(x) + sqrt(abs(x))
# The results are stored in the out EArray object, and the expression evaluation
# happens chunk-wise:

out = h5.create_earray('/', 'out',
                       atom = tb.Float64Atm(),
                       shape(0,n))

out.size_on_disk # should be 0

expr = tb.Expr('3 * sin(ear) + sqrt(abs(ear))') # Transforms str into Expr Object
expr.set_output(out, append_mode=True) # Defines output to be EArray object

expr.eval() # Initiates the evaluation of the expression -> %time this

out.size_on_disk # should be ~150m bytes

out_ = out.read() # Reads the whole EArray into memory -> %time this

# We notice that it is quite fast (for out-of-memory task)

# =============================================================================
# Can use in memory numexpr module for slightly aster results
# =============================================================================

import numexpr as ne

expr = '3 * sin(out_) + sqrt(abs(out_))'

ne.set_num_threads(1) # Sets number of threads to one

ne.evaluate(expr)[0, :10] # Evaluates expression using 1 thread -> %time this

ne.set_num_threads(4)

ne.evaluate(expr)[0, :10] # %time this -> Quicker than 1 thread

h5.close()