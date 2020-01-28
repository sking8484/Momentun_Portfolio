import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from iexfinance.stocks import get_historical_data
import datetime
from tqdm import tqdm
import warnings
key = 'sk_6d1c2037a984473895a42a17710cf794'
stock_list = pd.read_excel('sp500_const.xlsx')

stock_array = np.array(stock_list['STOCK_LIST'])


warnings.filterwarnings("ignore")

import seaborn as sns
end =datetime.datetime.now()
start = end - datetime.timedelta(365*5)

sns.set_style("darkgrid")

import matplotlib.dates as mdates












"""GET DATA FROM IEX"""






# pricing_data = pd.DataFrame(index=stock_data.index)
# end =datetime.datetime.now()
# start = end - datetime.timedelta(365*5)
# for x in stock_list.values:
#     stock_data = get_historical_data(x[0], start = start, end = end, token = key, close_only=True)
#     stock_data = pd.DataFrame(stock_data).T[['close']]
#     stock_data.columns = [x[0]]
#     pricing_data = pd.concat([pricing_data, stock_data], axis = 1)





"""GET PRICING DATA FROM PICKLE IF UNAVAILABLE FROM IEX"""

pricing_data= pd.read_pickle('pricing_data')
pricing_data.index = pd.to_datetime(pricing_data.index)



"""CREATE DATAFRAMES"""

momentum_columns = []
for x in pricing_data.columns:
    momentum_name = x + '_momentum'
    momentum_columns.append(momentum_name)

r2_columns = []
for x in pricing_data.columns:
    r2_name = x + '_r2'
    r2_columns.append(r2_name)

"""OPTIMIZE ALGORITHM, IF NECESSARY"""






# results_data = pd.DataFrame(columns = ["CUMULATIVE_RETURNS",
#            "WINDOW",
#            "SWITCH"])


# for window in np.arange(5,50,5):
#     for switch in np.arange(0,3,.2):



#         #create dataframes
#         momentum_data = pd.DataFrame(index = pricing_data.index, columns=momentum_columns)
#         r2_data = pd.DataFrame(index = pricing_data.index, columns = r2_columns)
#         trade_signals = pd.DataFrame(index = pricing_data.index)
#         pct_change_data = pd.DataFrame(index = pricing_data.index)
#         returns_data = pd.DataFrame(index = pricing_data.index)


#         momentum_window = window
#         momentum_switch = switch

#         i = 0
#         cut = 10
#         for stock in pricing_data.columns[::cut]:
#             stock_pricing = pricing_data[stock]
#             for day in np.arange(momentum_window,len(pricing_data.index),momentum_window):

#                 x_data = np.arange(momentum_window)
#                 y_data = stock_pricing[day-momentum_window:day]
#                 x_data = sm.add_constant(x_data)
#                 model = sm.OLS(y_data, x_data)
#                 res = model.fit()

#                 momentum_data.iat[day, i] = res.params.values[1]

#                 r2_data.iat[day,i] =res.rsquared




#             i = i+1


#         momentum_data.ffill(inplace = True, axis = 0)
#         r2_data.ffill(inplace = True, axis = 0)
#         momentum_data.fillna(0,inplace = True)
#         r2_data.fillna(0,inplace = True)


#             #change up the momentum_switch



#         for stock in pricing_data.columns[::cut]:
#             trade_signals[stock + '_signal'] = np.where(momentum_data[stock+'_momentum'] >= momentum_switch, 1, 0)
#             r2_data[stock + '_r2'] = r2_data[stock+'_r2']*trade_signals[stock + '_signal']
#             pct_change_data[stock + '_pct_change'] = pricing_data[stock].pct_change()
#         r2_data = r2_data.apply(normalize_r2, axis = 1)
#         for stock in pricing_data.columns[::cut]:

#             returns_data[stock + '_returns'] = pct_change_data[stock + '_pct_change'].shift(1) * r2_data[stock + '_r2']







#     #         for stock in pricing_data.columns:
#     #             trade_signals[stock + '_signal'] = np.where(momentum_data[stock+'_momentum'] >= momentum_switch, 1, 0)
#     #             r2_data[stock + '_r2'] = r2_data[stock+'_r2']*trade_signals[stock + '_signal']
#     #             pct_change_data[stock + '_pct_change'] = pricing_data[stock].pct_change()
#     #             returns_data[stock + '_returns'] = pct_change_data[stock + '_pct_change'].shift(1) * r2_data[stock + '_r2']

#         def normalize_r2(row):
#             new_row = row/sum(row)

#             return new_row





#         cum_data = returns_data.cumsum(axis = 0)
#         cum_data['daily_returns'] = cum_data.sum(axis = 1)
#         cum_data['daily_returns_cum'] = cum_data['daily_returns'].cumsum()


#         end_df = pd.DataFrame.from_dict({"CUMULATIVE_RETURNS":[cum_data['daily_returns_cum'].iloc[-1]],
#            "WINDOW":[window],
#            "SWITCH":[switch]})


#         results_data = results_data.append(end_df)
#         print(results_data)






"""RUN ALGORITHM AND CREATE TRADE SIGNALS"""





#create dataframes
momentum_data = pd.DataFrame(index = pricing_data.index, columns=momentum_columns)
r2_data = pd.DataFrame(index = pricing_data.index, columns = r2_columns)
trade_signals = pd.DataFrame(index = pricing_data.index)
pct_change_data = pd.DataFrame(index = pricing_data.index)
returns_data = pd.DataFrame(index = pricing_data.index)


momentum_window = 5
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


momentum_data.ffill(inplace = True, axis = 0)
r2_data.ffill(inplace = True, axis = 0)
r2_data.fillna(0, inplace = True)
momentum_data.fillna(0, inplace = True)


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




for stock in pricing_data.columns[::cut]:
    trade_signals[stock + '_signal'] = np.where(momentum_data[stock+'_momentum'] > momentum_switch, 1, 0)
    r2_data[stock + '_r2'] = r2_data[stock+'_r2']*trade_signals[stock + '_signal']

    pct_change_data[stock + '_pct_change'] = pricing_data[stock].pct_change()


def normalize_r2(row):
    new_row = row/sum(row)

    return new_row
r2_data = r2_data.apply(normalize_r2, axis=1)
for stock in pricing_data.columns[::cut]:
    returns_data[stock + '_returns'] = pct_change_data[stock + '_pct_change'].shift(1) * r2_data[stock + '_r2']








#         for stock in pricing_data.columns:
#             trade_signals[stock + '_signal'] = np.where(momentum_data[stock+'_momentum'] >= momentum_switch, 1, 0)
#             r2_data[stock + '_r2'] = r2_data[stock+'_r2']*trade_signals[stock + '_signal']
#             pct_change_data[stock + '_pct_change'] = pricing_data[stock].pct_change()
#             returns_data[stock + '_returns'] = pct_change_data[stock + '_pct_change'].shift(1) * r2_data[stock + '_r2']









cum_data = returns_data
cum_data['daily_returns'] = cum_data.sum(axis = 1)
cum_data['daily_returns_cum'] = cum_data['daily_returns'].cumsum()


trades(r2_data)
stat_plot()




# end_df = pd.DataFrame.from_dict({"CUMULATIVE_RETURNS":[cum_data['daily_returns_cum'].iloc[-1]],
#    "WINDOW":[window],
#    "SWITCH":[switch]})


# results_data = results_data.append(end_df)




def trades():
    portfolio_value = 1000000#get from robinhood

    try:
        orders = pd.read_excel('orders.xlsx')
    except Exception as e:
        print(e)
        orders = pd.DataFrame(columns = r2_data.columns)
        orders.to_excel('orders.xlsx')



    try:
        trading_data = pd.read_excel('trading_data.xlsx')
    except Exception as e:
        print(e)
        trading_data = pd.DataFrame(columns = r2_data.columns)
        trading_data.to_excel('trading_data.xlsx')


    positions_to_buy = pd.DataFrame(r2_data.iloc[-1]).T

    positions_to_buy = positions_to_buy.apply(find_shares, axis = 1)





    if len(trading_data.index) < 1:
        trading_data = pd.concat([trading_data, positions_to_buy])
        orders = pd.concat([orders, trading_data])

    else:


        trading_data = trading_data.append(positions_to_buy)
        orders = pd.concat([orders,positions_to_buy - trading_data.iloc[-1]])


    trading_data.to_excel('trading_data.xlsx')
    orders.to_excel('orders.xlsx')

    return orders










def find_shares(row):
    portfolio_value = 10000000 #get from robinhood
    trades = []
    for x in range(len(row)):
        mvalue = row[x]*portfolio_value
        shares = mvalue/pricing_data.iloc[-1][x]
        trades.append(shares)

    return trades




'''Create Trades'''

new_orders = trades()

for symbol, order in  zip(new_orders.iloc[-1].index, new_orders.iloc[-1]):
    if order > 0:
        print("BUY" + symbol[:-3] + ", " + str(order) + " shares")




"""CREATE STATISTICS"""


def stat_plot():
    number_positions = pd.DataFrame(index = pricing_data.index, columns = ['POSITIONS'])
    for x in range(len(number_positions.index)):

        row=trade_signals.iloc[x]
        num_pos = len(row[row>0])

        number_positions.iat[x,0] = num_pos

    stock = 'AAPL'

    portfolio = cum_data[cum_data.columns[-1]]
    stock_info = (pricing_data[stock]/pricing_data[stock].iloc[0] -1)


    stock_info.pct_change().dropna()[1:].std()
    portfolio_vol = portfolio.pct_change().dropna()[1:].std()

    spy = pd.DataFrame(get_historical_data("SPY", start= start, end = end, token = key, close_only=True)).T['close']
    spy.index = pd.to_datetime(spy.index)
    spy_vol = spy.pct_change().std()




    spy.pct_change().std()

    vol_dict= {"AAPL": [stock_info.pct_change().dropna()[1:].std()],"SPY":[spy_vol], "PORTFOLIO":[portfolio_vol]}

    vol_df = pd.DataFrame.from_dict(vol_dict)
    vol_df.index =  ['VOLATILITY']

    returns_dict = {"AAPL": [ stock_info.pct_change().dropna()[1:].mean()],"SPY":[ spy.pct_change().mean()], "PORTFOLIO":[ portfolio.pct_change().dropna()[1:].mean()]}
    ret_df= pd.DataFrame.from_dict(returns_dict)
    ret_df.index = ['RETURNS']



    positions = pd.DataFrame(columns = ['POSITION', 'PERCENTAGE'], index = pricing_data.index)
    for x in range(len(pricing_data.index)):
        positions.iat[x,0] = r2_data.iloc[x][:-2].sort_values(ascending = False)[:1].index[0][:-3]
        positions.iat[x,1] = r2_data.iloc[x][:-2].sort_values(ascending = False)[0]



    positions.dropna(inplace = True)
    positions['PERCENTAGE'] = positions['PERCENTAGE'].astype(float)

    #positions.groupby(["POSITION"]).mean().sort_values('PERCENTAGE',ascending= True)[-100:].plot.bar(figsize = (20,10))

    r2_data.iloc[25][:-2].sort_values(ascending = False)[0]
    portfolio.pct_change().dropna()[1:].std()
    #positions.plot.bar()


    fig,axes = plt.subplots(1, sharex=True)
    fig.autofmt_xdate()





    (pricing_data['AAPL']/pricing_data['AAPL'].iloc[0] - 1).plot( ax = axes, figsize = (20,10))
    portfolio.plot( ax=axes)
    (spy/spy.iloc[0] - 1).plot( ax = axes)
    axes.legend(['AAPL', 'PORTFOLIO', 'SPY'])

    fig,axes = plt.subplots(2, sharex=True)
    fig.autofmt_xdate()





    #axe.xaxis.set_minor_locator(mdates.MonthLocator())
    positions.plot( ax = axes[0], figsize = (20,5))


    number_positions.plot( ax = axes[1])
    axes[1].legend(['NUMBER OF POSITIONS'])



    plt.legend()

    fig, axes = plt.subplots(1, 2)

    vol_df.plot.bar(figsize = (20,5), ax =axes[0])
    ret_df.plot.bar(figsize = (20,5), ax = axes[1])





    fig, axe = plt.subplots()




    #plt.figure(figsize = (20,10))


    todays_holdings = r2_data.loc[r2_data.index[-1]]

    todays_100_largest = todays_holdings.sort_values(ascending=False)
    #todays_100_largest = todays_100_largest/todays_100_largest.sum()
    todays_100_largest.plot.pie(figsize = (20,10))
