import requests, re, json, pprint

p = re.compile(r'root\.App\.main = (.*);')
tickers = ['AAPL']
results = {}

# import sys
# from PyQt5.QtWidgets import QApplication, QWidget


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     w = QWidget()
#     w.resize(300,300)
#     w.setWindowTitle('Testing this title')
#     w.show()
#     sys.exit(app.exec_())

# with requests.Session() as s:

    # for ticker in tickers:
    #     r = s.get('https://finance.yahoo.com/quote/{}/key-statistics?p={}'.format(ticker,ticker))
    #     data = json.loads(p.findall(r.text)[0])
    #     key_stats = data['context']['dispatcher']['stores']['QuoteSummaryStore']
    #     res = {
    #             'Enterprise Value' : key_stats['defaultKeyStatistics']['enterpriseValue']['fmt']
    #             ,'Trailing P/E' : key_stats['summaryDetail']['trailingPE']['fmt']
    #             ,'Forward P/E' : key_stats['summaryDetail']['forwardPE']['fmt']
    #             ,'PEG Ratio (5 yr expected)' : key_stats['defaultKeyStatistics']['pegRatio']['fmt']
    #             , 'Return on Assets' : key_stats['financialData']['returnOnAssets']['fmt']
    #             , 'Quarterly Revenue Growth' : key_stats['financialData']['revenueGrowth']['fmt']
    #             , 'EBITDA' : key_stats['financialData']['ebitda']['fmt']
    #             , 'Diluted EPS' : key_stats['defaultKeyStatistics']['trailingEps']['fmt']
    #             , 'Total Debt/Equity' : key_stats['financialData']['debtToEquity']['fmt']
    #             , 'Current Ratio' :  key_stats['financialData']['currentRatio']['fmt']
    #     }
    #     results[ticker] = res

# pprint.pprint(results)

with requests.Session() as s:
    r = s.get('https://finance.yahoo.com/quote/AAPL?p=AAPL')
    data = json.loads(p.findall(r.text)[0])
    key_stats = data['context']['dispatcher']['stores']['QuoteSummaryStore']
    res = {
        'Volume' : key_stats['summaryDetail']['volume']['fmt'],
        'averageDailyVolume10Day' : key_stats['summaryDetail']['averageDailyVolume10Day']['fmt'],
        'marketCap' : key_stats['summaryDetail']['marketCap']['fmt']
        }
    
    results['AAPL'] = res
    
# pprint.pprint(results)
