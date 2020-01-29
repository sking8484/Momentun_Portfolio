from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from threading import Timer
import time



class TestApp(EClient, EWrapper):
    def __init__(self):
        EClient.__init__(self,self)
        #super(TestApp,self).__init__(self)
        self.account_val_dict = {}

        print("Initialized TestApp")
    def error(self, reqId, errorCode, errorString):
        print("ERROR: ", reqId, " ", errorCode , " ",errorString)

    def nextValidId(self, orderId):
        self.start()
    def updateAccountValue(self, key:str, val:str, currency:str,
                            accountName:str):




                            self.account_val_dict[key] = val
                            #print( self.account_val_dict)



    def start(self):
        print("STARTED")
        self.reqAccountUpdates(True, "DU1855372")

        #print(self.account_val_dict['NetLiquidation'])
    def get_liquidity(self):
        return self.account_val_dict['NetLiquidation']

    def stop(self):

        self.reqAccountUpdates(False, "DU1855372")
        self.done = True
        self.disconnect()

def main():
    app = TestApp()
    app.nextOrderId = 0
    app.connect("127.0.0.1", 7497, 0)


    Timer(5, app.stop).start()
    app.run()
    liquidity = app.get_liquidity()
    return liquidity
