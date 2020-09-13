import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates # matplotlib for some reason doesn't use datetime dates, so will have to use mdates
import pandas as pd
import pandas_datareader.data as web
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

style.use('ggplot')

################## HOW TO PULL DATA FROM YAHOO AND SAVE AS CSV ################## 
##start = dt.datetime(2014, 1, 1)
##end = dt.datetime(2018, 12, 31)
##df = web.DataReader('ACB', 'yahoo', start, end) # (ticker, source, start time, end time)
## print(df.head()) #df.head will print first 5 rows of dataframe -> can input a number if want or use df.tail to get the end
## adjusted close = adjusted for stock splits etc
##df.to_csv('tsla.csv') # converts it to a csv with name 'tsla.csv'

df = pd.read_csv('tsla.csv', parse_dates = True, index_col = 0)


df['100 SMA'] = df['Adj Close'].rolling(window=100, min_periods=0).mean()
                                # window = how many days prior to use, min_period makes it so if prev days don't exist, just uses what's available
#print(df.tail())
            # Won't show df.head() because at the start, there aren't 100 datapoints to pull from
df.dropna(inplace=True) # 'inplace = True' same thing as df = df.dropna
                        # above method removes anything that is NaN in df -> now can use df.head())
#print(df.head())

ax1 = plt.subplot2grid((6,1), (0,0), rowspan = 5, colspan = 1)
                                # first parameter is grid size (here 6 rows, 1 column)
                                # 2nd is starting point - since this is first graph, start at top corner (0,0)
                                # 3rd is how many rows it will span
                                # 4th is how many columns it will span

ax2 = plt.subplot2grid((6,1), (5,0), rowspan = 1, colspan = 1, sharex=ax1) #sharex will share x-axis with another dataset (i.e. when we zoom it, will zoom in on both graphs)

##################  RESAMPLING DATA TO REDUCE SIZE ##################
df_ohlc = df['Adj Close'].resample('10D').ohlc()
                                # Resampling to average every 10 days and show the OHLC
                                # OHLC = 'open high low close'
df_volume = df['Volume'].resample('10D').sum()

#candlestick_ohlc wants (mdate, date, open, high, low, close) in order to work
df_ohlc.reset_index(inplace=True) # adds another index column so that we are not using date as the index
df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num) # Converting date to mdates

ax1 = plt.subplot2grid((6,1),(0,0), rowspan = 5, colspan = 1)
ax2 = plt.subplot2grid((6,1),(5,0), rowspan = 1, colspan = 1, sharex = ax1)
ax1.xaxis_date() # converts mdates to actual dates

candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g', colordown='r')
    # (x-values, y-values, width of candlesticks, color for uptick, color for downtick)
ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
    # (x-values, y-values, start from this value)
    # fill_between for kind of 'area under curve' view instead of line
plt.show()



################## PLOTS MULTIPLE DATASETS ON GRAPH ################## 
##ax1.plot(df.index, df['Adj Close']) #index is the dates (x-axis), adj close is y-axis
##ax1.plot(df.index, df['100 SMA'])
##ax2.bar(df.index, df['Volume']) 
##
##plt.show()
