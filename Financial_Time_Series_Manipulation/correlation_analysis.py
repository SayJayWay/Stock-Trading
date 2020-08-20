# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 18:34:45 2020

@author: John
"""

import pandas as pd
import numpy as np

AAPL_data = pd.read_csv(r'../stock_dfs/AAPL.csv', index_col = 0, parse_dates = True)
AMD_data = pd.read_csv(r'../stock_dfs/AMD.csv', index_col = 0, parse_dates = True)

data = pd.concat([AAPL_data['Adj Close'], AMD_data['Adj Close']], axis=1, keys=['AAPL_data', 'AMD_data']).dropna()

# Separate subplots
data.plot(subplots=True, figsize = (10,6))
# Together on one plot
data.plot(secondary_y = 'AMD_data', figsize = (10,6))

# =============================================================================
# Logarithmic return
# =============================================================================
rets = np.log(data / data.shift(1))
rets.dropna(inplace = True)
rets.plot(subplots = True, figsize = (10,6))

# =============================================================================
# Scatter Matrix
# =============================================================================
pd.plotting.scatter_matrix(rets,
                           alpha = 0.2, # This is opacity
                           diagonal = 'hist',
                           hist_kwds = {'bins': 35},
                           figsize = (10,6))

# =============================================================================
# Ordinary Least-Squares Regression
# =============================================================================
reg = np.polyfit(rets['AAPL_data'], rets['AMD_data'], deg = 1)
ax = rets.plot(kind = 'scatter', x='AAPL_data', y='AMD_data', figsize = (10,6))
ax.set_xlabel('AAPL')
ax.set_ylabel('AMD')
ax.plot(rets['AAPL_data'], np.polyval(reg, rets['AAPL_data']), 'r', lw=2)
# Resulting graph shows positive line indicating support for positive correlation


# =============================================================================
# Correlation
# =============================================================================
rets.corr()
# Note -> Have to input following line separately
ax = rets['AAPL_data'].rolling(window=252).corr(rets['AMD_data']).dropna().plot(figsize = (10,6)) 
ax.axhline(rets.corr().iloc[0,1], c = 'r')

