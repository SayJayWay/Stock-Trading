#! Retrieve historical data from IB_insync

from time import sleep

import nest_asyncio
nest_asyncio.apply()

from ib_insync import *



def main():
    ib = IB()
    ib.connect('127.0.0.1', port = 7497, clientId = 556)
    print('CONNECTED')
    import datetime as dt
    start = dt.datetime(2019,6,20)
    end = datetime.datetime.now()
    barsList = []
    
    reqHistoricalData(contract, endDateTime, durationStr, barSizeSetting, 
                      whatToShow, useRTH, formatDate=1, keepUpToDate=False, 
                      chartOptions=[], timeout=60)

    dt = end
    
    contract = Stock('PLTR', 'SMART','USD')
    
    while dt > start:
        bars = ib.reqHistoricalData(
                contract,
                endDateTime=dt,
                durationStr='600 S',
                barSizeSetting='30 secs',
                whatToShow='MIDPOINT',
                useRTH=True,
                formatDate=1)
        barsList.append(bars)
        dt = bars[0].date
        
    allBars = [b for bars in reversed(barsList) for b in bars]
    df = util.df(allBars)
    f = open('hist.csv','a')
    f.write(str(df) )
    print(df)
    ib.disconnect()
    
if __name__ == '__main__':
    ib, trade = main()
    ib.disconnect()

'''contract (Contract) – Contract of interest.

endDateTime (Union[datetime, date, str, None]) – Can be set to ‘’ to indicate the current time, or it can be given as a datetime.date or datetime.datetime, or it can be given as a string in ‘yyyyMMdd HH:mm:ss’ format. If no timezone is given then the TWS login timezone is used.

durationStr (str) – Time span of all the bars. Examples: ‘60 S’, ‘30 D’, ‘13 W’, ‘6 M’, ‘10 Y’.

barSizeSetting (str) – Time period of one bar. Must be one of: ‘1 secs’, ‘5 secs’, ‘10 secs’ 15 secs’, ‘30 secs’, ‘1 min’, ‘2 mins’, ‘3 mins’, ‘5 mins’, ‘10 mins’, ‘15 mins’, ‘20 mins’, ‘30 mins’, ‘1 hour’, ‘2 hours’, ‘3 hours’, ‘4 hours’, ‘8 hours’, ‘1 day’, ‘1 week’, ‘1 month’.

whatToShow (str) – Specifies the source for constructing bars. Must be one of: ‘TRADES’, ‘MIDPOINT’, ‘BID’, ‘ASK’, ‘BID_ASK’, ‘ADJUSTED_LAST’, ‘HISTORICAL_VOLATILITY’, ‘OPTION_IMPLIED_VOLATILITY’, ‘REBATE_RATE’, ‘FEE_RATE’, ‘YIELD_BID’, ‘YIELD_ASK’, ‘YIELD_BID_ASK’, ‘YIELD_LAST’.

useRTH (bool) – If True then only show data from within Regular Trading Hours, if False then show all data.

formatDate (int) – For an intraday request setting to 2 will cause the returned date fields to be timezone-aware datetime.datetime with UTC timezone, instead of local timezone as used by TWS.

keepUpToDate (bool) – If True then a realtime subscription is started to keep the bars updated; endDateTime must be set empty (‘’) then.

chartOptions (List[TagValue]) – Unknown.

timeout (float) – Timeout in seconds after which to cancel the request and return an empty bar series. Set to 0 to wait indefinitely.'''
