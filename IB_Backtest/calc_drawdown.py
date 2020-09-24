#%% Risk Analysis

equity = 3333
risk = pd.DataFrame(test['lstrategy_tc_30']) # Relevant log returns time series...

# ...scaled by initial equity
risk['equity'] = risk['lstrategy_tc_30'].cumsum().apply(np.exp) * equity

risk['cummax'] = risk['equity'].cummax() # cumulative maximum values over time

risk['drawdown'] = risk['cummax'] - risk['equity'] # Drawdown values over time

risk['drawdown'].max() # Maximum drawdown value
#Out: 714.0806803769715

t_max = risk['drawdown'].idxmax() # Point in time when it happens
# Output : Timestamp('2018-06-29 02:45:00')

# Technically a (new) high is characterized by a drawdown value of 0.  The
# drawdown period is the time between two such highs

temp = risk['drawdown'][risk['drawdown'] == 0] # Identifies highs for which drawdown must be 0

# Calculates timedelta values between all highs
periods = (temp.index[1:].to_pydatetime() - temp.index[:-1].to_pydatetime())

# Longest drawdown period in seconds
t_per = periods.max()

# Longest drawdown period in hours
t_per.seconds / 60 / 60

risk[['equity', 'cummax']].plot(figsize = (10, 6))
plt.axvline(t_max, c = 'r', alpha = 0.5)
plt.title('Max drawdown (vertical line) and drawdown periods (horizontal lines)')