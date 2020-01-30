from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import *
from threading import Timer

class TestApp(EWrapper, EClient):
    def __init__(self, symbol, shares):
        EClient.__init__(self,self)
        self.symbol = symbol
        self.shares = shares

    def error(self, reqId, errorCode, errorString):
        print("ERROR: ", reqId, " ", errorCode, " ", errorString)

    def nextValidId(self, orderId):
        self.nextOrderId = orderId
        self.start(self.symbol, self.shares)

    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, permId, parentdId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        print("OrderStatus Id: ", orderId, "Status: ", status, 'filled: ',filled, 'Remaining: ', remaining, 'lastFillPrice: ', lastFillPrice)

    def openOrder(self, orderId, contract, order, orderState):
        print("OpenOrder ID: ", orderId, contract.symbol, contract.secType, "0", contract.exchange, ":", order.action, order.orderType, order.totalQuantity, orderState.status)

    def execDetails(self, reqId, contract, execution):
        print("ExecDetails: ", reqId, contract.symbol, contract.secType, contract.currency, execution.execId, execution.orderId, execution.shares, execution.lastLiquidity)




    def start(self, ticker, shares):


        contract = Contract()
        contract.symbol = ticker
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        contract.primaryExchange = "SMART"



        order = Order()
        order.action = "BUY"
        order.totalQuantity = shares
        order.orderType = "MKT"


        self.placeOrder(self.nextOrderId, contract, order)


    def stop(self):
        self.done = True
        self.disconnect()



def main(stock, shares):
    app = TestApp(stock, shares)
    app.nextOrderId = 0
    app.connect("127.0.0.1", 7497, 9)

    Timer(3,app.stop).start()
    app.run()

if __name__ == "__main__":
    trade_dict = {"AAPL":50,
                "TSLA":25,
                "GE":200}
    for key in trade_dict:
        main(key, trade_dict[key])
