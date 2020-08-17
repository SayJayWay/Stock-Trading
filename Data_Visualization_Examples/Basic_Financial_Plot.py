import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cufflinks as cf
import plotly.offline as plyo

raw = pd.read_csv('../stock_dfs/AAPL.csv', index_col = 0, parse_dates = True)

raw.info()

quotes = raw[['Open', 'High', 'Low', 'Close']]
quotes.iloc[-60:]

qf = cf.QuantFig(
            quotes,
            title = 'AAPL Stock',
            legend = 'top',
            name = 'AAPL Stock'
)

plyo.offline.plot(
            qf.iplot(asFigure = True),
            image = 'png',
            filename = './images/qf_01.html'
)

# Adding Bollinger Bands
qf.add_bollinger_bands(periods = 15,
                        boll_std = 2) # Number of standard deviations away

plyo.offline.plot(
            qf.iplot(asFigure = True),
            image = 'png',
            filename = './images/qf_02.html'
)

# Adding RSI
qf.add_rsi(periods = 14,
            showbands = False)

plyo.offline.plot(
            qf.iplot(asFigure = True),
            image = 'png',
            filename = './images/qf_03.html'
)
