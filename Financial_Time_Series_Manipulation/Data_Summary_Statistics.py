import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

data = pd.read_csv(r'../stock_dfs/AAPL.csv', index_col=0)

# Summary Statistics
aggregation = data.aggregate([min,
                            np.mean,
                            np.std,
                            np.median,
                            max]).round(2)

# print(aggregation)

# Daily differences (flat value)
daily_diff = data.diff().head()

# % Daily differences
percent_daily_diff = data.pct_change().round(3).head()

# Log daily returns (this has some form of normalization)
log_return = np.log(data / data.shift(1))
log_return.cumsum().apply(np.exp).plot(figsize=(10,6))

# plt.show()

# Data Resampling

data.index = pd.to_datetime(data.index) # Converts index to datetime objects
data_resampled = data.resample('1w', label='right').last().head()
print(data_resampled)
