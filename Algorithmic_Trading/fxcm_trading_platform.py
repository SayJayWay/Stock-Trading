import fxcmpy

# Setting up api to connect to your account
def connect():
    api = fxcmpy.fxcmpy(access_token='6b4421a5085b15448a105b2ab38987397db0ceb6', log_level = 'error')
    #api = fxcmpy.fxcmpy(config_file = 'fxcm.cfg', server = 'demo')



import time
import numpy as np
import pandas as pd
import datetime as dt
from pylab import plt, mpl

plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

from fxcmpy import fxcmpy_tick_data_reader as tdr

# Show all available currency pairs for which tick data is availbale
print(tdr.get_available_symbols())

def retrieve_data():
    start = dt.datetime(2018, 6, 25)
    stop = dt.datetime(2018,6 ,30)
    
    # Retrieve data file, unpack it, and store raw data in df object
    td = tdr('EURUSD', start, stop) 
    
    # returns df object w/ raw data (i.e. index values still str objects)
    td.get_raw_data().info()
    
    # returns df object where index is DatetimeIndex
    td.get_data().info
    return td

sub = td.get_data(start = '2018-06-29 12:00:00',
                  end = '2018-06-29 12:14:00') # Picks subset of complete data

sub['Mid'] = sub.mean(axis = 1)
sub['SMA'] = sub['Mid'].rolling(1000).mean()
sub[['Mid', 'SMA']].plot(figsize = (10, 6), lw = 0.75)

def retrieve_candle_data():
    from fxcmpy import fxcmpy_candles_data_reader as cdr
    print(cdr.get_available_symbols())
    
    # Similar data retrieval as regular data, just that a "period value" (i.e.
    # the bar length) needs to be specified
    
    start = dt.datetime(2018, 5, 1)
    stop = dt.datetime(2018,6, 30)
    
    period = 'H1' # period value (m1, m5, m15, m30, H1, H2, H3, H4, H6, H8, D1, W1, M1)
    
    candles = cdr('EURUSD', start, stop, period)
    return candles

candles = retrieve_candle_data()
data = candles.get_data()
data.head()
data.columns # (['BidOpen', 'BidHigh', 'BidLow', 'BidClose', 'AskOpen', 'AskHigh',
             # 'AskLow', 'AskClose'],
data[data.columns[:4]].tail() # OHLC for bid prices
data[data.columns[4:]].tail() # OHLC for ask prices

# Mid close prices from bid/ask close prices
data['MidClose'] = data[['BidClose', 'AskClose']].mean(axis = 1)

# 30SMA and 100SMA
data['SMA1'] = data['MidClose'].rolling(30).mean()
data['SMA2'] = data['MidClose'].rolling(100).mean()

data[['MidClose', 'SMA1', 'SMA2']].plot(figsize=(10, 6))

#%%
# =============================================================================
# Working with API
# =============================================================================
# Before was getting data from servers, another way is using the api that we 
# defined earlier

print(api.get_instruments())

def retrieve_api_data():
    candles = api.get_candles('USD/JPY', period = 'D1', number = 10)
    return candles

candles = retrieve_api_data()

candles[candles.columns[:4]]
candles[candles.columns[4:]]

start = dt.datetime(2017, 1, 1)
stop = dt.datetime(2018, 1, 1)

candles1 = api.get_candles('EUR/GBP', period = 'D1', start = start, stop = end)

# This will retrieve the most recent one-minute bar prices available
candles2 = api.get_candles('EUR/USD', period = 'm1', number = 250)

plt.figure(figsize = (10, 6))
candles2['askclose'].plot()
plt.title('Historical ask close prices for EUR/USD (minute bars)')

#%%
# =============================================================================
# Retrieving Streaming Data
# =============================================================================
# FXCM api allows for subscription to real-time data streams for all instruments
# fxcmpy wrapper package supports this functionality, in that it allows users
# to provide user-defined functions ("callback functions") to process the real-
# time data stream

# Example callback function that prints out selected elements of data set retrived
def output(data, dataframe):
    print('%3d | %s | %s | %6.5f, %6.5f'
          % (len(dataframe), data['Symbol'],
             pd.to_datetime(int(data['Updated']), unit = 'ms'),
             data['Rates'][0], data['Rates'][1]))

# Subscription to specific real-time data stream; data is processed async as long
    # as there is no 'unsubscribe' event
api.subscribe_market_data('EUR/USD', (output,))

api.get_last_price('EUR/USD') # Gets last available data set during a subscription

api.unsubscribe_market_data('EUR/USD')

#%%
# =============================================================================
# Placing Orders
# =============================================================================

api.get_open_positions() #  Shows open positions for connected (default) account

# Opens 100,000 position in EUR/USD currency pair
order = api.create_market_buy_order('EUR/USD', 10)

# Show open positions for selected elements only
sel = ['tradeId', 'amountK', 'currency', 'grossPL', 'isBuy'] 
api.get_open_positions()[sel]

# Opens 50,000 position in EUR/GBP currency pair
order = api.create_market_buy_order('EUR/GBP', 5)

# Reduces position in EUR/USD
order = api.create_market_sell_order('EUR/USD', 3)

# Increases position in EUR/GBP
order = api.create_market_buy_order('EUR/GBP', 5)

# For EUR/GBP there are now two open long positions; contrary to EUR/USD position
# they are not netted
api.get_open_positions()[sel]

# Closses all positions for specified symbol
api.close_all_for_symbol('EUR/GBP')

# Closes all positions
api.close_all()

#%%
# =============================================================================
# Account Information
# =============================================================================
# Shows default accountId value
api.get_default_account()

# Shows for all accounts the financial situation and some parameters
api.get_accounts().T