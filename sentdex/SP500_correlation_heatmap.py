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

style.use('ggplot')

def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, features="lxml")
        # resp.txt is the text of source code
        # 'lxml' is a parser
    table = soup.find('table', {'class':'wikitable sortable'}) # find table w/ this class name
    tickers=[]

    for row in table.findAll('tr')[1:71]: #for each table row (tr) except the header ([1:])
        ticker = row.findAll('td')[1].text # need .text since right now it is a soup object
        tickers.append(ticker)
    with open("sp500tickers.pickle", "wb") as f: # wb = 'write bytes' -> makes it editable
        pickle.dump(tickers, f) # dumping tickers to file f (saves it to f)
    print(tickers)
    return tickers

##save_sp500_tickers()

def get_data_from_yahoo(reload_sp500=False): #set reload_sp500 to false
    
    if reload_sp500: #If reload_sp500 is true, rerun the prev function
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)

    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2016,1,1)
    end = dt.datetime(2017,4,14)

    for ticker in tickers[100:102]:
        print(ticker)
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = web.DataReader(ticker, 'yahoo', start, end)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))


##get_data_from_yahoo()

def compile_data():
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame() # creates a data frame object with no columns, no index, etc
    print(tickers)

    for count, ticker in enumerate(tickers): # enumerate lets us count through things, ex: range(len(x)) etc
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)

        df.rename(columns = {'Adj Close':ticker}, inplace=True) # Since it is a dictionary, to rename we use a colon instead of a comma
        df.drop(['Open','High','Low','Close','Volume'], 1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer') # how='outer' let's us have information from both excels
                                                    # (adds on only on outside, doesn't change anything that already exists)
                                                    # May get NaN if the company did not exist on stock market at start date
                                                    # how = 'inner' only keeps common items
                                                    # Think of it like Venn Diagram -> inner VS outer of circle

        if count % 10 == 0:
            print(count)

    print(main_df.head())
    main_df.to_csv('sp500_joined_closes.csv')

##compile_data()

def visualize_data():
    df = pd.read_csv('sp500_joined_closes.csv')
##    df['AAPL'].plot()
##    plt.show()
    df_corr = df.corr() # creates a correlation table of our dataframe
                        # will compare all the values and calculate all the correlation values
                        # Can use this to see how much the companies move in unison -> to create a very diverse profile

    print(df_corr.head())
    
    data = df_corr.values # Gets the values from the table w/o the headers
    fig = plt.figure()
    ax = fig.add_subplot(111) # means 1 by 1, and it is plot number 1 -> can also do .add_subplot(1, 1, 1)

    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn) #RdYlGn makes the heat map from Rd = Red, Yl = Yellow, to Gn = Green
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False) # setting x-axis ticks at every half marks
                                                                # np.arange will return evenly spaced values within a given interval
                                                                # minor = false make all ticks minor and show each one
                                                                # minor = True will make bigger major ticks at intervals (ex: at 5, 10, 15, 20, etc)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.invert_yaxis() # removes gap at top of matplotlib graph
    ax.xaxis.tick_top() # moves x-axis ticks from bottom to top

    column_labels = df_corr.columns
    row_labels = df_corr.index

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation = 90) # rotates the labels of each company 90 degrees so it looks nicer
    heatmap.set_clim(-1,1) # sets the range of the heat map
    plt.tight_layout() # helps clean out layout
    plt.show()

visualize_data()
