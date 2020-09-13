import bs4 as bs
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import os
import pandas as pd
import pandas_datareader.data as web
import pickle # serializes any python object
import requests
##from TestAlgo import createData, save_obj, load_obj

##createData('GOOG')
##data = load_obj('data/Goog')
##
##t = float("{:.6f}".format(time.time()))
##for number, value in enumerate(data['Open'][0:len(data['Open'])-4]):
##	sum(data['Open'][number: number+4])
##elapsed = "{:.6f}".format(time.time()-t)
##print(elapsed)


start = dt.datetime(2017,1,1)
end = dt.datetime(2019,11,30)

tickers = ['SPY', 'GLD', 'MSFT', 'TSLA']

for ticker in tickers:
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
           df = web.DataReader(ticker, 'yahoo', start, end)
           df.to_csv('stock_dfs/{}.csv'.format(ticker))
           print('success')
        else:
           print('Already have {}'.format(ticker))

