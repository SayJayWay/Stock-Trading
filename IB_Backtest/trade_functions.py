# from ib.opt import Connection, message
# from ib.ext.Contract import Contract
# from ib.ext.Order import Order
from time import sleep

import nest_asyncio
nest_asyncio.apply()

from ib_insync import *

#%% This is if using ib-native
# def make_contract(symbol, sec_type, exch, prim_exch, curr):
#     Contract.m_symbol = symbol
#     Contract.m_secType = sec_type
#     Contract.m_exchange = exch
#     Contract.m_primaryExchange = prim_exch
#     Contract.m_currency = curr
#     return Contract

# def make_order(action, quantity, price = None):
#     if price is not None:
#         order = Order()
#         order.m_orderType = 'LMT'
#         order.m_totalQuantity = quantity
#         order.m_action = action # Buy or Sell
#         order.m_lmtPrice = price
#     else:
#         order = Order()
#         order.m_orderType = 'MKT'
#         order.m_totalQuantity = quantity
#         order.m_action = action # Buy or Sell    
#     return order

#%% 
def main():
    ib = IB()
    ib.connect('127.0.0.1', port = 7497, clientId = 556)
    print('CONNECTED')
    
    # ib-native method
    # cont = make_contract('TSLA', 'STK', 'SMART', 'SMART', 'USD')
    # offer = make_order('BUY', 1, 100)
    
    # ib_insync method  
    contract = Stock('AAPL', 'SMART', 'USD')
    ib.qualifyContracts(contract)
    order = MarketOrder('SELL', 1) 
    trade = ib.placeOrder(contract, order)
    
    # ib.placeOrder(contract = cont, order = offer)
    sleep(1)
    print('ORDER PLACED')
    ib.disconnect()
    return ib, trade
    
if __name__ == '__main__':
    ib, trade = main()
    ib.disconnect()