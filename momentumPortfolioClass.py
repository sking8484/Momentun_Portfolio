import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from iexfinance.stocks import get_historical_data
import datetime
from tqdm import tqdm
import warnings

import seaborn as sns


sns.set_style("darkgrid")

import matplotlib.dates as mdates

warnings.filterwarnings("ignore")



class momentum_algo():
    def __init__(self, optimize, place_trades):

        self.stock_list = pd.read_excel('sp500_const.xlsx')

        self.stock_array = np.array(self.stock_list['STOCK_LIST'])

        self.end =datetime.datetime.now()
        self.start = self.end - datetime.timedelta(365*5)
        self.key = 'sk_6d1c2037a984473895a42a17710cf794'
        self.optimize = optimize
        self.place_trades = place_trades

    def update_data(self):



        """GET DATA FROM IEX"""



        try:
            pricing_data= pd.read_pickle('pricing_data')
            pricing_data.index = pd.to_datetime(pricing_data.index)
        except exception as e:
            print("Couldn't find pickle")
            pricing_data = pd.DataFrame()
            end =datetime.datetime.now()
            start = end - datetime.timedelta(365*5)
            for x in self.stock_list.values:
                try:
                    stock_data = get_historical_data(x[0], start = start, end = end, token = key, close_only=True)
                    stock_data = pd.DataFrame(stock_data).T[['close']]
                    stock_data.columns = [x[0]]
                    pricing_data = pd.concat([pricing_data, stock_data], axis = 1)
                except exception as e:
                    print(e)
                    pass
            pricing_data.to_pickle('pricing_data')

        """UPDATE DATA"""

        temp_df = pd.DataFrame()


        if pricing_data.iloc[-1].name.day != datetime.datetime.today().day-1:
            for x in pricing_data.columns:
                stock_data = get_historical_data(x, start = self.end - datetime.timedelta(1), end = self.end, token = self.key, close_only=True)
                stock_data = pd.DataFrame(stock_data).T[['close']]
                stock_data.columns = [x]
                temp_df = pd.concat([temp_df, stock_data], axis = 1)
                print("UPDATED: " + x)


            pricing_data = pd.concat([pricing_data,temp_df], axis =0)
            pricing_data.to_pickle('pricing_data')
        self.pricing_data = pricing_data


        self.create_columns()









    def create_columns(self):
        """CREATE DATAFRAMES"""

        self.momentum_columns = []
        for x in self.pricing_data.columns:
            momentum_name = x + '_momentum'
            self.momentum_columns.append(momentum_name)

        self.r2_columns = []
        for x in self.pricing_data.columns:
            r2_name = x + '_r2'
            self.r2_columns.append(r2_name)

        """OPTIMIZE ALGORITHM, IF NECESSARY"""
        if self.optimize:
            self.optimizer()
        else:
            self.create_signals()



    def optimizer(self):


        results_data = pd.DataFrame(columns = ["CUMULATIVE_RETURNS",
                   "WINDOW",
                   "SWITCH"])


        for window in np.arange(5,6):
            for switch in np.arange(0,1,.001):



                #create dataframes
                momentum_data = pd.DataFrame(index = self.pricing_data.index, columns=self.momentum_columns)
                r2_data = pd.DataFrame(index = self.pricing_data.index, columns = self.r2_columns)
                trade_signals = pd.DataFrame(index = self.pricing_data.index)
                pct_change_data = pd.DataFrame(index = self.pricing_data.index)
                returns_data = pd.DataFrame(index = self.pricing_data.index)


                momentum_window = window
                momentum_switch = switch

                i = 0
                cut = 10
                for stock in self.pricing_data.columns[::cut]:
                    stock_pricing = self.pricing_data[stock]
                    for day in np.arange(momentum_window,len(self.pricing_data.index),momentum_window):

                        x_data = np.arange(momentum_window)
                        y_data = stock_pricing[day-momentum_window:day]
                        x_data = sm.add_constant(x_data)
                        model = sm.OLS(y_data, x_data)
                        res = model.fit()

                        momentum_data.iat[day, i] = res.params.values[1]

                        r2_data.iat[day,i] =res.rsquared




                    i = i+1


                momentum_data.ffill(inplace = True, axis = 0)
                r2_data.ffill(inplace = True, axis = 0)
                momentum_data.fillna(0,inplace = True)
                r2_data.fillna(0,inplace = True)


                    #change up the momentum_switch

                def top_momentum(row):
                    top_number = 50
                    threshold = row.sort_values(ascending=False)[top_number]
                    new_row = []
                    for weight in row:
                        if weight > threshold:
                            new_row.append(weight)
                        else:
                            new_row.append(0)
                    return new_row

                momentum_data = momentum_data.apply(top_momentum, axis = 1)




                for stock in self.pricing_data.columns[::cut]:
                    trade_signals[stock + '_signal'] = np.where(momentum_data[stock+'_momentum'] >= momentum_switch, 1, 0)
                    r2_data[stock + '_r2'] = r2_data[stock+'_r2']*trade_signals[stock + '_signal']
                    pct_change_data[stock + '_pct_change'] = self.pricing_data[stock].pct_change()
                r2_data = r2_data.apply(normalize_r2, axis = 1)
                for stock in pricing_data.columns[::cut]:

                    returns_data[stock + '_returns'] = pct_change_data[stock + '_pct_change'].shift(-1) * r2_data[stock + '_r2']







            #         for stock in pricing_data.columns:
            #             trade_signals[stock + '_signal'] = np.where(momentum_data[stock+'_momentum'] >= momentum_switch, 1, 0)
            #             r2_data[stock + '_r2'] = r2_data[stock+'_r2']*trade_signals[stock + '_signal']
            #             pct_change_data[stock + '_pct_change'] = pricing_data[stock].pct_change()
            #             returns_data[stock + '_returns'] = pct_change_data[stock + '_pct_change'].shift(1) * r2_data[stock + '_r2']

                def normalize_r2(row):
                    new_row = row/sum(row)

                    return new_row





                cum_data = returns_data
                cum_data['daily_returns'] = cum_data.sum(axis = 1)
                cum_data['daily_returns_cum'] = cum_data['daily_returns'].cumsum()


                end_df = pd.DataFrame.from_dict({"CUMULATIVE_RETURNS":[cum_data['daily_returns_cum'].iloc[-1]],
                   "WINDOW":[window],
                   "SWITCH":[switch]})


                results_data = results_data.append(end_df)
                print(results_data)









    #create dataframes
    def create_signals(self):
        """RUN ALGORITHM AND CREATE TRADE SIGNALS"""
        pricing_data = self.pricing_data
        momentum_data = pd.DataFrame(index = pricing_data.index, columns=self.momentum_columns)
        r2_data = pd.DataFrame(index = pricing_data.index, columns = self.r2_columns)
        trade_signals = pd.DataFrame(index = pricing_data.index)
        pct_change_data = pd.DataFrame(index = pricing_data.index)
        returns_data = pd.DataFrame(index = pricing_data.index)


        momentum_window = 3
        momentum_switch = 0

        i = 0
        cut = 1
        for stock in pricing_data.columns[::cut]:
            stock_pricing = pricing_data[stock]
            for day in np.arange(momentum_window,len(pricing_data.index),momentum_window):

                x_data = np.arange(momentum_window)
                y_data = stock_pricing[day-momentum_window:day]
                x_data = sm.add_constant(x_data)
                model = sm.OLS(y_data, x_data)
                res = model.fit()

                momentum_data.iat[day, i] = res.params.values[1]

                r2_data.iat[day,i] =res.rsquared




            i = i+1
            if i%25 == 0:
                print(str(i) + "/" + str(len(pricing_data.columns)))


            momentum_data.ffill(inplace = True, axis = 0)
            r2_data.ffill(inplace = True, axis = 0)
            r2_data.fillna(0, inplace = True)
            momentum_data.fillna(0, inplace = True)


                #change up the momentum_switch


            def top_momentum(row):
                top_number = 10
                threshold = row.sort_values(ascending=False)[top_number]
                new_row = []
                for weight in row:
                    if weight > threshold:
                        new_row.append(weight)
                    else:
                        new_row.append(0)
                return new_row

            #momentum_data = momentum_data.apply(top_momentum, axis = 1)




            for stock in pricing_data.columns[::cut]:
                trade_signals[stock + '_signal'] = np.where(momentum_data[stock+'_momentum'] > momentum_switch, 1, 0)
                r2_data[stock + '_r2'] = r2_data[stock+'_r2']*trade_signals[stock + '_signal']

                pct_change_data[stock + '_pct_change'] = pricing_data[stock].pct_change()


            def normalize_r2(row):
                new_row = row/sum(row)

                return new_row
            r2_data = r2_data.apply(normalize_r2, axis=1)
            for stock in pricing_data.columns[::cut]:
                returns_data[stock + '_returns'] = pct_change_data[stock + '_pct_change'].shift(-1) * r2_data[stock + '_r2']






            cum_data = returns_data
            cum_data['daily_returns'] = cum_data.sum(axis = 1)
            cum_data['daily_returns_cum'] = cum_data['daily_returns'].cumsum()


            self.r2_data = r2_data

            self.cum_data = cum_data
        if self.place_trades():
            self.trades()




    def trades(self):
        portfolio_value = 1000000#get from robinhood

        try:
            orders = pd.read_excel('orders.xlsx')
        except Exception as e:
            print(e)
            orders = pd.DataFrame(columns = self.r2_data.columns)
            orders.to_excel('orders.xlsx')



        try:
            trading_data = pd.read_excel('trading_data.xlsx')
        except Exception as e:
            print(e)
            trading_data = pd.DataFrame(columns = self.r2_data.columns)
            trading_data.to_excel('trading_data.xlsx')


        positions_to_buy = pd.DataFrame(self.r2_data.iloc[-1]).T

        positions_to_buy = positions_to_buy.apply(self.find_shares, axis = 1)





        if len(trading_data.index) < 1:
            trading_data = pd.concat([trading_data, positions_to_buy])
            orders = pd.concat([orders, trading_data])

        else:
            trading_data = trading_data.append(positions_to_buy)
            orders = pd.concat([orders,positions_to_buy - trading_data.iloc[-2]])


        trading_data.to_excel('trading_data.xlsx')
        orders.to_excel('orders.xlsx')

        self.print_orders(orders)










    def find_shares(self, row):
        portfolio_value = 10000000 #get from robinhood
        trades = []
        for x in range(len(row)):
            mvalue = row[x]*portfolio_value
            shares = mvalue/pricing_data.iloc[-1][x]
            trades.append(shares)

        return trades




    '''Create Trades'''


    def print_orders(self,new_orders):

        order_dict = {}
        for symbol, order in  zip(new_orders.iloc[-1].index, new_orders.iloc[-1]):
            if order > 1:

                order_dict[symbol] = [order, "BUY"]
            if order <-1:
                order_dict[symbol]= [order, "SELL"]
        self.stat_plot()
        return order_dict





    def stat_plot(self):
        number_positions = pd.DataFrame(index = self.pricing_data.index, columns = ['POSITIONS'])
        for x in range(len(number_positions.index)):

            row=trade_signals.iloc[x]
            num_pos = len(row[row>0])

            number_positions.iat[x,0] = num_pos

        stock = 'AAPL'

        portfolio = (self.cum_data[self.cum_data.columns[-1]] + 1)
        stock_info = (self.pricing_data[stock]/self.pricing_data[stock].iloc[0])


        stock_info.pct_change().dropna()[1:].std()
        portfolio_vol = portfolio.pct_change().dropna()[1:].std()

        spy = pd.DataFrame(get_historical_data("SPY", start= start, end = end, token = key, close_only=True)).T['close']
        spy.index = pd.to_datetime(spy.index)





        spy_vol=spy.pct_change().dropna()[1:].std()

        vol_dict= {"AAPL": [stock_info.pct_change().dropna()[1:].std()],"SPY":[spy_vol], "PORTFOLIO":[portfolio_vol]}

        vol_df = pd.DataFrame.from_dict(vol_dict)
        vol_df.index =  ['VOLATILITY']

        returns_dict = {"AAPL": [ stock_info.pct_change().dropna()[1:].mean()],"SPY":[ spy.pct_change().dropna()[1:].mean()], "PORTFOLIO":[ portfolio.pct_change().dropna()[1:].mean()]}
        ret_df= pd.DataFrame.from_dict(returns_dict)
        ret_df.index = ['RETURNS']

        vol_ret_df = pd.concat([vol_df, ret_df])

        sharpe_df = pd.DataFrame(vol_ret_df.loc['RETURNS']/vol_ret_df.loc['VOLATILITY'])

        sharpe_df.columns = ['SHARPE']
        sharpe_df = sharpe_df.T



    #     sharpe_df = pd.concat([vol_ret_df, sharpe_df])
    #     sharpe_df = pd.DataFrame(sharpe_df.loc['SHARPE'])




        positions = pd.DataFrame(columns = ['POSITION', 'PERCENTAGE'], index = self.pricing_data.index)
        for x in range(len(self.pricing_data.index)):
            positions.iat[x,0] = self.r2_data.iloc[x][:-2].sort_values(ascending = False)[:1].index[0][:-3]
            positions.iat[x,1] = self.r2_data.iloc[x][:-2].sort_values(ascending = False)[0]



        positions.dropna(inplace = True)
        positions['PERCENTAGE'] = positions['PERCENTAGE'].astype(float)

        #positions.groupby(["POSITION"]).mean().sort_values('PERCENTAGE',ascending= True)[-100:].plot.bar(figsize = (20,10))

        r2_data.iloc[25][:-2].sort_values(ascending = False)[0]
        portfolio.pct_change().dropna()[1:].std()
        #positions.plot.bar()


        fig,axes = plt.subplots(1, sharex=True)
        fig.autofmt_xdate()





        (self.pricing_data['AAPL']/self.pricing_data['AAPL'].iloc[0] ).plot( ax = axes, figsize = (20,10))
        portfolio.plot( ax=axes)
        (spy/spy.iloc[0] ).plot( ax = axes)
        axes.legend(['AAPL', 'PORTFOLIO', 'SPY'])

        fig,axes = plt.subplots(2, sharex=True)
        fig.autofmt_xdate()





        #axe.xaxis.set_minor_locator(mdates.MonthLocator())
        positions.plot( ax = axes[0], figsize = (20,5))


        number_positions.plot( ax = axes[1])
        axes[1].legend(['NUMBER OF POSITIONS'])



        plt.legend()

        fig, axes = plt.subplots(1,3)


        vol_df.plot.bar(figsize = (20,5), ax =axes[0])
        ret_df.plot.bar(figsize = (20,5), ax = axes[1])
        sharpe_df.plot.bar(figsize = (20,5), ax = axes[2])





        fig, axe = plt.subplots()




        #plt.figure(figsize = (20,10))


        todays_holdings = self.r2_data.loc[self.r2_data.index[-1]]

        todays_100_largest = todays_holdings.sort_values(ascending=False)
        #todays_100_largest = todays_100_largest/todays_100_largest.sum()
        todays_100_largest.plot.pie(figsize = (20,10))

    def run(self):
        self.update_data()
