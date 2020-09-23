from ib.opt import Connection, message
from ib.ext.Contract import Contract
from ib.ext.Order import Order

def make_contract(symbol, sec_type, exch, prim_exch, curr):
    Contract.m_symbol = symbol
    Contract.m_secType = sec_type
    Contract.m_exchange = exch
    Contract.m_primaryExchange = prim_exch
    Contract.m_currency = curr
    return Contract

def make_order(action, quantity, price = None):
    if price is not None:
        order = Order()
        order.m_orderType = 'LMT'
        order.m_totalQuantity = quantity
        order.m_action = action # Buy or Sell
        order.m_lmtPrice = price
    else:
        order = Order()
        order.m_orderType = 'MKT'
        order.m_totalQuantity = quantity
        order.m_action = action # Buy or Sell
        
    return order

def main():
    conn = Connection.create(port = 7497, clientId = 556)
    print('CONNECTED')
    conn.connect()
    
    oid = 500
    cont = make_contract('TSLA', 'STK', 'SMART', 'SMART', 'USD')
    offer = make_order('BUY', 1, 100)
    
    conn.placeOrder(oid, cont, offer)
    print('ORDER PLACED')
#    conn.disconnect()
#    print('DISCONNECTED')
    
if __name__ == '__main__':
    main()