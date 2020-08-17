import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.preprocessing import MinMaxScaler

def save_obj(data, filename):
    with open('pickle/' + filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load_obj(filename):
    with open('pickle/' + filename, 'rb') as f:
        return pickle.load(f)

def computeRatio(data):
    data['RCO'] = []    # Close:Open
    data['RHO'] = []    # High:Open
    data['RHC'] = []    # High:Close
    data['RHA'] = []    # High:Adj Close
    data['ROL'] = []    # Open:Low
    data['RCL'] = []    # Close:Low
    data['RAL'] = []    # Adj Close:Low
    for number, value in enumerate(data['Open']): # create f'n to check if len of all is same?  This could fail
        data['RCO'].append((data['Close'][number]-data['Open'][number])/data['Close'][number] * 100)
        data['RHO'].append((data['High'][number]-data['Open'][number])/data['High'][number] * 100)
        data['RHC'].append((data['High'][number]-data['Close'][number])/data['High'][number] * 100)
        data['RHA'].append((data['High'][number]-data[' Adj Close'][number])/data['High'][number] * 100)
        data['ROL'].append((data['Open'][number]-data['Low'][number])/data['Open'][number] * 100)
        data['RCL'].append((data['Close'][number]-data['Low'][number])/data['Close'][number] * 100)
        data['RAL'].append((data['High'][number]-data['Adj Close'][number])/data['High'][number] * 100)

    return data

def createData(csv):
    df1 = pd.read_csv("stock_dfs/" + csv + ".csv")
    df = df1.drop('Date', 1) # remove date as it is irrelevant
    data = df.to_dict()
    colmean = []
    colstd = []

    standardized_df = (df-df.mean())/df.std()
    normalized_df = (df-df.min())/(df.max()-df.min())

    standdata = standardized_df.to_dict()
    normdata = normalized_df.to_dict()


    datadict = {}
    standdict = {}
    normdict = {}
    
    for header in data.keys():
        datadict[header] = datadict.setdefault(header,[])
        standdict[header] = standdict.setdefault(header,[])
        normdict[header] = normdict.setdefault(header,[])
        for value in data[header]:
            datadict[header].append(data[header][value])
        for value in standdata[header]:
            standdict[header].append(standdata[header][value])
        for value in normdata[header]:
            normdict[header].append(normdata[header][value])
    data = datadict
    standdata = standdict
    normdata = normdict

    data = computeRatio(data)
    
    save_obj(data, 'data/' + csv)
    save_obj(standdata, 'standdata/' + csv)
    save_obj(normdata, 'normdata/' + csv)

createData('GOOG')
df = pd.read_csv("stock_dfs/" + csv + ".csv")
data = load_obj('data/GOOG')
normdata = load_obj('normdata/GOOG')
standdata = load_obj('standdata/GOOG')

# Plotting
ax1 = plt.subplot2grid((6,1),(0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1),(5,0), rowspan=1, colspan=1, sharex=ax1)
ax1.plot(df1.index, data['Close'])  # Uses Date from dataframe and data from data
ax2.bar(df1.index, data['Volume'])  # 

plt.show()
# Mean of ratios within last 'x' time period to see general trend
# Support and Resistance line (using highs/lows and creating tangent line) -- problem might not know where to have starting point...otherwise will miss out on a lot of lines
# Calculate VWAP, EMA,
def computeVWAP():
    vwap = sum()
    return VWAP
# Calculate ratio bins between size of body and size of tails (i.e. shooting stars something like 5:1...4:1, etc) to trial and error from

