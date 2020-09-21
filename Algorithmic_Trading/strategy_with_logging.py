import zmq
import time
import pickle
import fxcmpy
import numpy as np
import pandas as pd
import datetime as dt

sel = ['tradId', 'amountK', 'currency', 'grossPL', 'isBuy']

log_file = 'automated_strategy.log'

# loads the persisted algorithm object
algorithm = pickle.load(open('algorithm.pkl', 'rb'))

# sets up the socket communication via ZeroMQ (here: "publisher")
context = zmq.Context()
socket = context.socket(zmq.PUB)

# this binds the socket communication to all IP addresses of the machine
socket.bind('tcp://0.0.0.0:5555')

def logger_monitor(message, time = True, sep = True):
    ''' Custom logger and monitor function.
    '''
    
    with open(log_file, 'a') as f:
        t = str.datetime.now()
        msg = ''
        if time:
            msg += '\n' + t + '\n'
        if sep:
            msg += 66 * '=' + '\n'
        msg += message + '\n\n'
        # sends the message via the socket
        socket.send_string(msg)
        # writes the message to the log file
        f.write(msg)
        
def report_positions(pos):
    ''' Prints, logs and sends position data.
    '''
    out = '\n\n' + 50 * '=' + '\n'
    out += 'Going {}.\n'.format(pos) + '\n'
    time.sleep(2) # waits for the order to be executed
    out += str(api.get_open_positions()[sel]) + '\n'
    out += 50 * '=' + '\n'
    logger_monitor(out)
    print(out)
    
def automated_strategy(data, dataframe):
    ''' Callback function embodying the trading logic.
    '''
    global min_bars, position, df
    # resampling of tick data
    df = dataframe.resample(bar, label = 'right').last().ffill()
    
    if len(df) > min_bars:
        min_bars = len(df)
        logger_monitor('NUMBER OF TICKS: {} | '.format(len(dataframe)) + 
                       'NUMBER OF BARS: {}'.format(min_bars))
        # data processing and feature preparation
        df['Mid'] = df[['Bid', 'Ask']].mean(axis = 1)
        df['Returns'] = np.log(df['Mid'] / df['Mid'].shift(1))
        df['Direction'] = npwhere(df['Returns'] > 0, 1, -1)
        # picks relevant points
        features = df['Direction'].iloc[-(lags + 1):-1]
        #necessary reshaping
        features = features.values.reshape(1, -1)
        # generates the signal (+1 or -1)
        signal = algorithm.predict(features)[0]
        
        # logs and sends major financial information
        logger_monitor('MOST RECENT DATA\n' + 
                       str(df[['Mid', 'Returns', 'Direction']].tail()), False)
        logger_monitor('features: ' + str(features) + '\n' +
                       'position: ' + str(position) + '\n' +
                       'signal:   ' + str(signal), False)
        
        # trading logic
        if position in [0, -1] and signal == 1: # going long?
            # Places a buy order
            api.create_market_buy_order(symbol, size - position * size) 
            position = 1 # Changes position to long
            report_positions('LONG')
            
        elif position in [0, 1] and signal == -1: # going short?
            # Places a sell order
            api.create_market_sell_order(symbol, size + position * size)
            position = -1 # Changes position to short
            report_positions('SHORT')
        else: # no trade
            logger_monitor('No trade placed')
            
        logger_monitor('****END OF CYCLE***\n\n', False, False)
        
    if len(dataframe) > 350: # stopping condition
        api.unsubscribe_market_data('EUR/USD') 
        report_positions('CLOSE OUT')
        api.close_all() # closes all open positions
        logger_monitor('***CLOSING OUT ALL POSITIONS***')
        
if __name__ = '__main__':
    symbol = 'EUR/USD' # symbol to be traded
    bar = '15s' # bar length; adjust for testing/deployment
    size = 100 # position size in thousand currency units
    position = 0 # initial position 
    lags = 5 # number of lags for features data
    min_bars = lags + 1 # minimum length for resampled DataFrame
    df = pd.DataFrame()
    # adjust configuration file location
    api = fxcmpy.fxcmpy(config_file = '../fxcm.cfg')
    # the main asynchronous loop using the callback function
    api.subscribe_market_data(symbol, (automated_strategy,))
    
#%% Strategy Monitoring
# Used to implement a local or remote monitoring of the algo trading strategy
# via socket communication
    
import zmq

# sets up the socket communication via ZeroMQ (here: "subscriber")
context = zmq.Context()
socket = context.socket(zmq.SUB)

# adjust the IP address to reflect the remote location
socket.connect('tcp://REMOTE_IP_ADDRESS:5555')

# CONFIGURES THE SOCKET TO RETRIEVE EVERY MESSAGE
socket.setsockopt_string(zme.SUBSCRIBE, '')

while True:
    msg = socket.recv_string()
    print(msg)