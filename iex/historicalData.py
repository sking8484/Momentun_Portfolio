import requests
import json
import pandas as pd


class iexFinance:

    def __init__(self, token):

        print("Initialized IEX API CONNECTION")
        self.token = token
        self.base = 'https://cloud.iexapis.com/stable/stock/'

    def get_historical_chart(self,symbol,date):
        
        response = requests.get(self.base + symbol+'/' +'chart' + '/' +date+'?chartCloseOnly=True&token=' + self.token )
        dict_list = json.loads((response.content).decode('utf-8'))
        parent_set = pd.DataFrame()
        for observation in dict_list:
            child = pd.DataFrame.from_dict([observation])
            child.set_index(child.date, inplace = True)
            child.index = pd.to_datetime(child.index)
            child.drop(columns = ['date'], inplace = True)
            child = child[['close']]
            child.columns = [symbol]

            parent_set = pd.concat([parent_set,child], axis = 0)

        return parent_set
