from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt
ts = TimeSeries(key='WFJ5OP8E2OF56H55',output_format='pandas')
data, meta_data = ts.get_intraday(symbol='AMD',interval='1min', outputsize='full')
#print(data)
data.to_csv('AMDalpha_vantage.csv')
#data['4. close'].plot()
#plt.title('Intraday TimeSeries Google')
#plt.show()

