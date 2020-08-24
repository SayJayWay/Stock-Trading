# TsTables uses PyTables for high-performance storage for time-series data.
# Useful for "write once, retrieve multiple times" scenarios.  This is common
# in financial analystics, where data is created in markets, retrieved in 
# real-time of asynchronously, and stored on disk for usage.

import numpy as np
import pandas as pd
import tables as tb
import tstables as tstab
import datetime as dt

# =============================================================================
# Data Preparation
# =============================================================================

no = 5000000 # number of time steps
co = 3 # number of time series
interval = 1. / (12 * 30 * 24 * 60) # time interval as a year fraction
vol = 0.2 # Volatility

# %%time
rn = np.random.standard_normal((no, co))
rn[0] = 0.0 # Sets initial random numbers to 0

# Simulation based on Euler discretization
paths = 100 * np.exp(np.cumsum(-0.5 * vol ** 2 * interval +
                               vol * np.sqrt(interval) * rn, axis=0))

paths[0] = 100 # Initial values of the paths is 100

dr = pd.date_range('2019-1-1', periods = no, freq='1s')
dr[-6:]

df = pd.DataFrame(paths, index=dr, columns=['ts1', 'ts2', 'ts3'])
df.info()

df[::100000].plot(figsize=(10,6))

# =============================================================================
# Data Storage with TsTables
# =============================================================================
# TsTables stores financiel time series data bas on chunk-based structure that
# allows for fast retrieval of arbitrary data subsets defined by some time
# interval.

class ts_desc(tb.IsDescription):
    timestamp = tb.Int64Col(pos=0) # Column for time stamps
    ts1 = tb.Float64Col(pos=1)
    ts2 = tb.Float64Col(pos=2)
    ts3 = tb.Float64Col(pos=3)
    
h5 = tb.open_file('tstab.h5','w')
ts = h5.create_ts('/', 'ts', ts_desc) # Creates TsTable object based on the ts_desc object

ts.append(df) # %time this -> Will produce error on latest pandas version

type(ts)

# =============================================================================
# Data Retrieval with TsTables
# =============================================================================

# TsTables returns a DataFrame object
read_start_dt = dt.datetime(2019, 2, 1, 0, 0)
read_end_dt = dt.datetime(2019, 2, 5, 23, 59)

rows = ts.read_range(read_start_dt, read_end_dt) # Returns df object for the interval -> %time this
rows.info()
h5.close()

(rows[::500] / rows.iloc[0]).plot(figsize=(10,6))

# =============================================================================
# Another example of Data Retrieval
# =============================================================================
# Retrieves 10 chunks of data consisting of 3 days worth of 1-second bars.
# DF has 345,600 rows of data and retrieval takes < 1/10 of a second

import random
h5 = tb.open_file('tstab.h5', 'r')
ts = h5.root.ts._f_get_timeseries() # Connects to the TsTable object

for _ in range(100): # %% time this -> ~80ms!
    d = random.randint(1, 24)
    read_start_dt = dt.datetime(2019, 2, d, 0, 0, 0)
    read_end_dt = dt.datetime(2019, 2, d + 3, 23, 59, 59)
    rows = ts.read_range(read_start_dt, read_end_dt)
