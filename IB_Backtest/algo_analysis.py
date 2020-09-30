#%% Risk Analysis
import pandas as pd
import numpy as np
from pylab import plt
import scipy.stats as scs
#

class Algo_Analysis:
    '''Algo_Analysis is a class created to perform various statistical analyses:
       
       Parameters
       ==========
       res_df: results dataframe that must contain the following columns:
                   - returns: benchmark log returns if one was to hold the stock
                              without any trading
                   - <strat_log_ret_column>: a column that contains the log returns
                                             of the strategy to be observed
    
       Analyses include:
                1) calc_Drawdown - Returns maximum drawdown value and drawdown
                                    period
                2) calc_Cvar - Returns credit value at risk for multiple confidence
                                intervals
                3) calc_Kelly - Returns Kelly Criterion for position sizing
    '''
    def __init__(self, res_df):
        self.res_df = res_df
    
    def calc_Drawdown(self, strat_log_ret_column, num_shares):        
        ''' Calculates maximum drawdown and drawdown period
        Parameters
        ===========
        strat_log_ret_column: Name of column that contains the log returns of the
                        algo/strategy
                        
        num_shares: Number of shares that should be purchased (can be determined
                    via Kelly Criterion --- Might remove and make it auto calculate)
        
        Returns
        =======
        t_per_seconds: Longest drawdown period in seconds
        
        t_per_hours: Longest drawdown period in hours
        
        max_drawdown: Maximum drawdown value
        '''
        risk = self.res_df # Relevant log returns time series...
        
        # ...scaled by initial equity...
        risk['equity'] = self.res_df[strat_log_ret_column].cumsum().apply(np.exp) * num_shares        
        risk['cummax'] = risk['equity'].cummax() # cumulative maximum values over time        
        risk['drawdown'] = risk['cummax'] - risk['equity'] # Drawdown values over time      
        self.max_drawdown = risk['drawdown'].max() # Maximum drawdown value
        
        t_max = risk['drawdown'].idxmax() # Point in time when it happens
        temp = risk['drawdown'][risk['drawdown'] == 0] # Identifies highs for which drawdown must be 0
        
        # Calculates timedelta values between all highs
        periods = (temp.index[1:].to_pydatetime() - temp.index[:-1].to_pydatetime())
        
        # Longest drawdown period in seconds
        self.t_per_seconds = periods.max()
        
        # Longest drawdown period in hours
        self.t_per_hours = self.t_per_seconds.seconds / 60 / 60
        
        risk[['equity', 'cummax']].plot(figsize = (10, 6))
        plt.axvline(t_max, c = 'r', alpha = 0.5)
        plt.title('Max drawdown (vertical line) and drawdown periods (horizontal lines)')
        
    def calc_Var(self, strat_log_ret_column, num_shares, hourly = False):
        '''Calculates Value-at-Risk for multiple confidence intervals
        Parameters
        ==========
        strat_log_ret_column: Name of column that contains the log returns of the
                        algo/strategy
        num_shares: Number of shares that should be purchased (can be determined
                    via Kelly Criterion --- Might remove and make it auto calculate)
        Returns
        =======
        Var: Returns Value-at-Risk values for various confidence levels
        '''
        percs = np.array([0.01, 0.1, 1. , 2.5, 5.0, 10.0]) # % percentages to be used
        
        risk = self.res_df[strat_log_ret_column]
        
        def print_var():
                print('%16s %16s' % ('Confidence level', 'Value-at-Risk'))
                print(33 * '-')
                for pair in zip(percs, self.Var):
                    print('%16.2f %16.3f' % (100 - pair[0], -pair[1]))
              
        if hourly = True:
            # Re-evaluating VaR values for a time horizon of 1-hour by resampling the original
            # DataFrame object.  In effect, VaR values are increased for all CIs but the
            # highest one.
            
            # Resamples data from 5-minute bars to 1-hr bars
            hourly = risk.resample('1H', label = 'right').last()
            hourly['returns'] = np.log(hourly['equity'] / hourly['equity'].shift(1))
            
            # Recalculate VaR values for resampled data
            self.Var = scs.scoreatpercentile(num_shares * hourly['returns'], percs)
        else:
            # Translates percent values to CIs and Var Values to positive values for printing
            self.Var = scs.scoreatpercentile(num_shares * risk, percs)
                
        print_var()
            
        
    def calc_Kelly(self, strat_log_ret_column, prob_success):
        ''' Calculates full Kelly and half Kelly values based on Kelly criterion
        
        Parameters
        ==========
        strat_log_ret_column: Name of column that contains the log returns of the
                        algo/strategy
                        
        prob_success: probability of success given in decimal value
        
        Returns
        =======
        full_Kelly: Optimal leverage according to Kelly criterion ("full Kelly")
        
        half_Kelly: full_Kelly * 0.5
        '''
        # Optimized Leverage
        # Since we have trading strategy's log returns data, the mean and variance 
        # values can be calculated in order to derive the optimal leverage according
        # to the Kelly criterion.  We can scale the numbers to annualized values:
        
        mean = self.res_df[['returns', strat_log_ret_column]].mean() * len(self.res_df) * 12 # Annualized mean returns
        
        var = self.res_df[['returns', strat_log_ret_column]].var() * len(self.res_df) * 12 # Annualized variances
        
        vol = var ** 0.5 # Annualized volatilities
        
        self.full_Kelly = mean / var # Optimal leverage according to Kelly criterion ("full Kelly")
        
        self.half_Kelly = mean / var * 0.5 # "half Kelly"
        
        # Using the "half Kelly" criterion, the optimal leverage for the trading 
        # strategy is about 40.  This leverage is possible in FX markets.
        
        to_plot = ['returns', strat_log_ret_column]
        
        for lev in [10, 20, 30, 40, 50]:
            label = '%s_%d' % (strat_log_ret_column, lev)
            self.res_df[label] = self.res_df[strat_log_ret_column] * lev # Scales strategy returns for different leverage values
            to_plot.append(label)
            
        self.res_df[to_plot].cumsum().apply(np.exp).plot(figsize = (10, 6))
        plt.title('Performance of algo trading strategy for different leverage values')
