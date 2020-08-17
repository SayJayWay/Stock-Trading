import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import cufflinks as cf
import plotly.offline as plyo

import numpy as np
plyo.init_notebook_mode(connected=True)

# Data Prepping
a = np.random.standard_normal((250, 5)).cumsum(axis=0)

index = pd.date_range('2019-1-1',
                        freq = 'B',
                        periods = len(a))

df = pd.DataFrame(100 + 5 * a,
                    columns = list('abcde'),
                    index = index)

# Graphing
plyo.offline.plot(
        df.iplot(asFigure = True),
        image = 'png',
        filename = './images/ply_01.html'
)

# Alternative Graphing
plyo.offline.plot(
        df[['a', 'b']].iplot(asFigure = True,
        theme = 'polar',
        title = 'A Time Series Plot',
        xTitle = 'Date',
        yTitle = 'Value',
        mode = {'a': 'markers', 'b' : 'lines+markers'},
        symbol = {'a': 'circle', 'b' : 'diamond'},
        size = 3.5,
        colors = {'a' : 'blue', 'b' : 'magenta'},
        ),
        image = 'png',
        filename = './images/ply_02.html'
)

# Bar Graphs
plyo.offline.plot(
        df.iplot(kind = 'hist',
            subplots = True,
            bins = 15,
            asFigure = True),
        image = 'png',
        filename = './images/ply_03.html'
)
