#%% Risk Analysis
import pandas as pd
import numpy as np
from pylab import plt
import scipy.stats as scs

class Algo_Analysis:
    def __init__(self, res_df):
        self.res_df = res_df
    
    def calc_Drawdown(self, log_ret_column, num_shares):        
        risk = self.res_df # Relevant log returns time series...
        
        # ...scaled by initial equity...
        risk['equity'] = self.res_df[log_ret_column].cumsum().apply(np.exp) * num_shares        
        risk['cummax'] = risk['equity'].cummax() # cumulative maximum values over time        
        risk['drawdown'] = risk['cummax'] - risk['equity'] # Drawdown values over time      
        risk['drawdown'].max() # Maximum drawdown value
        
        t_max = risk['drawdown'].idxmax() # Point in time when it happens
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
        
    def calc_Cvar(self, log_ret_column, num_shares):
        percs = np.array([0.01, 0.1, 1. , 2.5, 5.0, 10.0]) # % percentages to be used
        
        risk = self.res_df[log_ret_column]
        
        # VaR based on percentage values
        VaR = scs.scoreatpercentile(num_shares * risk, percs)
        
        # Translates percent values to CIs and Var Values to positive values for printing
        def print_var():
            print('%16s %16s' % ('Confidence level', 'Value-at-Risk'))
            print(33 * '-')
            for pair in zip(percs, VaR):
                print('%16.2f %16.3f' % (100 - pair[0], -pair[1]))
                
        print_var()
        
        # Re-evaluating VaR values for a time horizon of 1-hour by resampling the original
        # DataFrame object.  In effect, VaR values are increased for all CIs but the
        # highest one.
        
        # Resamples data from 5-minute bars to 1-hr bars
        hourly = risk.resample('1H', label = 'right').last()
        hourly['returns'] = np.log(hourly['equity'] / hourly['equity'].shift(1))
        
        # Recalculate VaR values for resampled data
        VaR = scs.scoreatpercentile(num_shares * hourly['returns'], percs)
        
        print_var()
        
    def calc_kelly(self, log_ret_column, prob_success):
        # Optimized Leverage
        # Since we have trading strategy's log returns data, the mean and variance 
        # values can be calculated in order to derive the optimal leverage according
        # to the Kelly criterion.  We can scale the numbers to annualized values:
        
        mean = test[['returns', log_ret_column]].mean() * len(self.res_df) * 12 # Annualized mean returns
        #returns       -0.040535
        #strategy_tc   -0.023228
        
        var = test[['returns', 'strategy_tc']].var() * len(self.res_df) * 12 # Annualized variances
        #returns        0.007861
        #strategy_tc    0.007855
        
        vol = var ** 0.5 # Annualized volatilies
        #returns        0.088663
        #strategy_tc    0.088630
        
        mean / var # Optimal leverage according to Kelly criterion ("full Kelly")
        #returns       -5.156448
        #strategy_tc   -2.956984
        
        mean / var * 0.5 # "half Kelly"
        #returns       -2.578224
        #strategy_tc   -1.478492
        
        # Using the "half Kelly" criterion, the optimal leverage for the trading 
        # strategy is about 40.  This leverage is possible in FX markets.
        
        to_plot = ['returns', 'strategy_tc']
        
        for lev in [10, 20, 30, 40, 50]:
            label = 'lstrategy_tc_%d' % lev
            test[label] = test['strategy_tc'] * lev # Scales strategy returns for different leverage values
            to_plot.append(label)
            
        test[to_plot].cumsum().apply(np.exp).plot(figsize = (10, 6))
        plt.title('Performance of algo trading strategy for different leverage values')