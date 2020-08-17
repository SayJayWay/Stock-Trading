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
        filename = './ply_01'
)
