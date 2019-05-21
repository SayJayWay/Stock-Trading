<h1>Stock Trading Scripts</h1>

<p>Python codes related to stock trading</p>
<p> Reference: https://www.youtube.com/watch?v=2BrpKpWwT2A </p>

<h2><b>Descriptions</b></h2>

<h3><b>7-day_intraday_alpha_vantage_stock_prices.py</b></h3>
<ul>
  <li> Retrieves intraday stock price for past 7 days (regular hours 9AM-4PM) and plots the closing price </li>
  <li> Note: Possible time intervals: "1min", "5min", "15min", "30min", "60min" </li>
  <li> Note: Possible outputsizes: "compact", "full" -- compact returns only the latest 100 data points in the intraday time series </li>
</ul>

<h3><b>SP500_correlation_heatmap.py</b></h3>
<ul>
  <li> save_sp500_tickers() - Retrieves list of SP500 companies from wikipedia using BeautifulSoup module and writes to pickle module </li>
  <li> get_data_from_yahoo(reload_sp500=False) - Retrieves data from pickle and downloads interday data from Yahoo Finance -> saves as CSV in folder named 'stock_dfs' </li>
  <li> compile_data() - Creates excel sheet with all the tickers adjusted closing prices from dates downloaded </li>
  <li> visualize_data() - Creates a heat map that correlates all the tickers to one another to see how each company moves in relation to each other </li>
</ul>

<h3>plot-stockdata-from-excel.py</h3>
<ul>
  <li> Retrieves interday data for specific ticker from Yahoo Finance </li>
  <li> Creates 100MA based on adjusted close and converts data to Open, high, low, close (OHLC) format</li>
  <li> Plots data with volume subplot </li>
</ul>

<h3> preprocessing_data_for_ML.py </h3>
<ul>
  <li> process_data_for_labels(ticker) - Pulls csv created in "SP500_correlation_heatmap.py" and calculates the percent change day to day for each stock </li>
  <li> buy_sell_hold(*args) - returns a boolean to determine if stock has moved at least 2% within the last day </li>
  <li> extract_featuresets(ticker) - Returns a normalized feature set of stocks as well as the buy/hold/sell classifications </li>
  <li> do_ml(ticker) - Uses a voting classifier to vote for best course of action and back tests against 25% of sample data.  Here, classifiers used are: Linear support vector classifier, k-nearest neighbours, random forest classifier.  Prints accuracy as well as predicted spread. </li>
</ul>
