from collections import Counter
import numpy as np
import pandas as pd
import pickle
from sklearn import svm, neighbors
from sklearn import model_selection as cross_validation
# svm = sport vector machine
# cross_validation to create nice training and testing samples because don't wanna test samples we trade against (want them to be blind)
# neighbors for 'k' nearest neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
# VotingClassifier to use many classifiers and let them vote on what they think is best -> to smooth out any 'unstable' classifications
# Random Forest Classifier -> just another classifier


def process_data_for_labels(ticker): # labels are your classifications/targets (i.e. buy, hold, sell)
    hm_days = 7
    df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True) # replaces NaN w/ specified value
    
    for i in range(1, hm_days+1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker] # shift will pull data from the future ('i' day)

    df.fillna(0, inplace=True)

    return tickers, df


process_data_for_labels('AAPL')

def buy_sell_hold(*args): #*args let's us input as many arguments as we want
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0

def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)
    df['{}_target'.format(ticker)] = list(map( buy_sell_hold,
                                               df['{}_1d'.format(ticker)],
                                               df['{}_2d'.format(ticker)],
                                               df['{}_3d'.format(ticker)],
                                               df['{}_4d'.format(ticker)],
                                               df['{}_5d'.format(ticker)],
                                               df['{}_6d'.format(ticker)],
                                               df['{}_7d'.format(ticker)],
                                               ))
                                        # list turns the entries into a mutable list
                                        # map will run a function (1st param) and execute it with the following iterables (other params)
                                        # This will return a list w/ the dates as index
    vals = df['{}_target'.format(ticker)].values.tolist() # this will return the values of the {}_target column w/o the date index
    str_vals = [str(i) for i in vals] # converts all values in "vals" to a string
    print('Data spread:', Counter(str_vals)) # Note that counter only works w/ strings and not integers
    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan) # replace infinity and -infinity with NaN
                                        # Some stocks percent changes may be inf/-inf if dividing by NaN (i.e. dealing w/ stocks that JUST got onto the exchang
    df.dropna(inplace=True)

    
    df_vals = df[[ticker for ticker in tickers]].pct_change() #.pct_change will calculate % change b/w current and prior (prev row) element
                                                                # We do % changes in order to normalize our data
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values # Our feature sets (normalized)
    y = df['{}_target'.format(ticker)].values # Our buy/hold/sell classifications

    return X, y, df

def do_ml(ticker):
    X, y, df = extract_featuresets(ticker)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.25)
                                                                            # 25% of our sample data will be what we test against (this will give us our accuracy)

##  clf = neighbors.KNeighborsClassifier() # Defines a SIMPLE classifier -> use votingclassifier to put multiple classifiers and let them vote which is best
                                            # Should know which classifiers are good for which applications
    clf = VotingClassifier([('lsvc', svm.LinearSVC()), # lsvc = Linear support vector classifier 
                            ('knn', neighbors.KNeighborsClassifier()), # knn = k nearest neighbours
                            ('for', RandomForestClassifier())]) # rfor = Random Forest


    
    clf.fit(X_train, y_train) # clf.fit is equivalent to .train
    confidence = clf.score(X_test, y_test)
    print('Accuracy:', confidence)
    predictions = clf.predict(X_test)
    print('Predicted spread:', Counter(predictions))

    return confidence

do_ml('BAC')
