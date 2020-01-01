# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 20:21:34 2018

@author: adein
"""

import numpy as np
np.random.seed(1)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import dataprovider as dp
import feature_generator as fg
from sklearn.externals import joblib
from os import listdir
from os.path import isfile, join




np.random.seed(20)

# Some options for configuration
FORECAST_HORIZON = 120            # for the targets in minutes
TRAIN_TEST_SPLIT = 2/3

""" (0) Load (and build) raw data """
path_to_raw_data = '../Exchanges/'
dataprovider = dp.CryptoMinuteDataProvider(path_to_raw_data)
raw_data = dataprovider.get_full_raw_data()



""" (1) Define features and targets """
feature_calculations = []
""" Multi-period returns """
for period in np.append(np.arange(1, 21), [120 * x for x in np.arange(1, 13)]):
    feature_calculations.append((fg.calculate_discrete_multiperiod_returns, {'period': period}))




""" (2) Compute features and targets """
features, targets = fg.compile_features_and_targets(raw_data, FORECAST_HORIZON, 
                                                    2, feature_calculations)



""" (3) Split the data into train and test set """
all_dates = raw_data.index.get_level_values('Date').drop_duplicates().values
idx_first_date_of_testing = int(len(all_dates)*TRAIN_TEST_SPLIT)
idx_first_date_of_training = idx_first_date_of_testing - FORECAST_HORIZON - 1
first_date_of_testing = all_dates[idx_first_date_of_testing]
last_date_of_training = all_dates[idx_first_date_of_training]

training_features = features.loc[features.index.get_level_values('Date') <= last_date_of_training]
training_targets = targets.loc[targets.index.get_level_values('Date') <= last_date_of_training]

test_features = features.loc[features.index.get_level_values('Date') >= first_date_of_testing]
test_targets = targets.loc[targets.index.get_level_values('Date') >= first_date_of_testing]


""" (4) Train the model """
model = RandomForestClassifier(n_estimators=1000, max_depth=20, n_jobs=-1, verbose=2)
model.fit(training_features, training_targets)
# in case we want to store it on disk, uncomment

#joblib.dump(model, 'rf.h5')


# to save upon memory, we predict coin by coin
coin_names = test_targets.index.get_level_values('Stock').drop_duplicates().values
predictions = []
for a_coin in coin_names:
    print('[INFO] Predicting coin', a_coin)
    coin_test_features = test_features.xs(a_coin, level='Stock')
    a_pred = model.predict_proba(coin_test_features)
    a_pred = a_pred[:, -1]
    a_pred = pd.DataFrame(a_pred)
    a_pred.index = test_features.xs(a_coin, level='Stock', drop_level=False).index
    a_pred.columns = ['prediction']
    predictions.append(a_pred)
predictions = pd.concat(predictions)
predictions = predictions.sort_index(0, level='Date')



""" (5) Perform the backtest """
# Retrieve trading volume at the timestamp when the position is opened
# (The opening date is the volume FORECAST_HORIZON minutes prior to closing the
# position)
def kpi_backtest(path_data, raw_data, staleness_filter, model_name, model_type='universal', modell='classification', tc=0.002):
    #import functions_new as functions
    
    path_result        = "C:/Users/adein/Anaconda3/envs/py35/Results/"
    
    print("[INFO] The KPI for a staleness filter of " + str(staleness_filter) + " and TC of " + str(tc) + " are being caluclated!")
    print(model_name)
    if model_type == 'universal':
        predictions = pd.read_csv(path_data)
        predictions.columns = ['Date', 'Stock', 'prediction']
        predictions.index = [pd.to_datetime(predictions.Date), predictions.Stock]
        del(predictions['Date'])
        del(predictions['Stock'])
    else:
        predictions = build_complete_df(path_data)
    
    FORECAST_HORIZON = 120
    opening_date_vol = raw_data.volume_from.unstack()
    opening_date_vol = opening_date_vol.shift(FORECAST_HORIZON-1)
    opening_date_vol_stacked = opening_date_vol.stack()
    opening_date_vol_stacked.name = 'opening_vol'
    
    
    # Compute the actual returns (we open at the open and close at the close of the respective bars)
    open_prices = raw_data.open.unstack()
    close_prices = raw_data.close.unstack()
    close_volume = raw_data.volume_from.unstack()
    
    # when no volume for closing is available, bfill with the first row for which volume is available
    for a_col in close_prices.columns:
        close_prices.loc[close_volume[a_col]<=1, a_col] = np.NAN
    close_prices = close_prices.bfill()
    
    actual_returns = close_prices/open_prices.shift(FORECAST_HORIZON-1) - 1
    actual_returns = actual_returns.stack()
    actual_returns.name = 'actual_returns'
    
    
    """ First join the returns to remove those timeseries that do not trade in the trading period """
    predictions = predictions.join(actual_returns)
    predictions = predictions.dropna()
    
    # extract the dates ---> needed for the offsets later on
    dates = list(predictions.index.get_level_values('Date').drop_duplicates())
    
    K = 5
    top_k = predictions.groupby(level='Date').prediction.nlargest(K)
    top_k.index = top_k.index.droplevel(0)
    top_k = pd.DataFrame(top_k)
    top_k = top_k.join(actual_returns)
    top_k.columns=['prediction', 'return_long']
    
    if modell == 'classification':
        top_k = top_k[top_k['prediction']>0.5]
    else:
        top_k = top_k[top_k['prediction']>0]
    
    
    flop_k = predictions.groupby(level='Date').prediction.nsmallest(K)
    flop_k.index = flop_k.index.droplevel(0)
    flop_k = pd.DataFrame(flop_k)
    flop_k = flop_k.join(actual_returns)
    flop_k.columns=['prediction', 'return_short']
    flop_k.return_short *= -1
    
    if modell == 'classification':
        flop_k = flop_k[flop_k['prediction']<0.5]
    else:
        flop_k = flop_k[flop_k['prediction']<0]
    
    
    # Compute the staleness/volume filter
    STALENESS = staleness_filter
    staleness_filter = opening_date_vol.shift(1) # we only use information up to one minute prior to opening the position
    staleness_filter = ((staleness_filter>=1)*1.0).rolling(window=1440).sum()/(1440)
    staleness_filter = staleness_filter.stack()
    staleness_filter = staleness_filter.loc[staleness_filter>=STALENESS]
    staleness_filter.name = 'staleness_filter'
    
    # Apply the staleness filter (we join and drop the items that do not make it through the filter)
    """ (Change Alex) Drop those of the top / flop 5 that do not make it through the filter """
    top_k = top_k.join(staleness_filter)
    top_k = top_k.dropna()
    del(top_k['staleness_filter'])
    
    flop_k = flop_k.join(staleness_filter)
    flop_k = flop_k.dropna()
    del(flop_k['staleness_filter'])
    
    # Average per trade returns just with staneless/volume filter
#    print('[INFO] With staleness filter of', STALENESS)
#    print(top_k.describe())
#    print(flop_k.describe())
    
    
    # Remove trades that have no volume in the opening minute
    top_k_with_vol = top_k.join(opening_date_vol_stacked)
    top_k_with_vol = top_k_with_vol.loc[top_k_with_vol.opening_vol > 1]
    flop_k_with_vol = flop_k.join(opening_date_vol_stacked)
    flop_k_with_vol = flop_k_with_vol.loc[flop_k_with_vol.opening_vol > 1]
    
#    print('[INFO] With staleness filter of', STALENESS, 'and opening volume and exit when volume is available')
#    print(top_k_with_vol['return_long'].describe())
#    print(flop_k_with_vol['return_short'].describe())
    
    
    # Subtract transaction costs and aggregate results
    """ Aggregate to daily returns """
    tops = top_k_with_vol.copy()
    tops['return_long'] -= tc
    tops['return_long'] *= (1.0/K) # to have a committed capital view (weight return)
    tops = tops['return_long'].groupby(level='Date').sum() # ... and sum()
    
    flops = flop_k_with_vol.copy()
    flops['return_short'] -= tc
    flops['return_short'] *= (1.0/K)
    flops = flops['return_short'].groupby(level='Date').sum()
    
    top_flop = pd.concat([tops, flops], axis=1)
    top_flop = top_flop.replace(np.NaN, 0)
    top_flop['return_total'] = top_flop.mean(axis=1)
    
    # aggregating before transaction costs
    tops_no_tc = top_k_with_vol.copy()
    tops_no_tc['return_long'] *= (1.0/K) # to have a committed capital view (weight return)
    tops_no_tc = tops_no_tc['return_long'].groupby(level='Date').sum() # ... and sum()
    
    flops_no_tc = flop_k_with_vol.copy()
    flops_no_tc['return_short'] *= (1.0/K)
    flops_no_tc = flops_no_tc['return_short'].groupby(level='Date').sum()
    
    top_flop_no_tc = pd.concat([tops_no_tc, flops_no_tc], axis=1)
    top_flop_no_tc = top_flop_no_tc.replace(np.NaN, 0)
    top_flop_no_tc['return_total'] = top_flop_no_tc.mean(axis=1)
    
    # plot daily returns for different offsets (0 minutes, 1 minute, 2 minutes, ...)
    dates_offset = [dates[offset::120] for offset in range(120)]
    
    all_means = []
    daily_rets = 0
    all_means_no_tc = []
    daily_rets_tc = 0
    
#    daily_rets_list = []
#    daily_rets_no_tc_list = []
    
    #import matplotlib.pyplot as plt
    n = 1
    #cherry_indices = [0,10,70,80,90,100,110]
    #for i in cherry_indices:
    for i in np.arange(0,120,n):
        #trades_top = top_k_with_vol[top_k_with_vol.index.get_level_values('Date').isin(dates_offset[0])]
        # only get those trades that fit in the offset scheme ---> trades that were executed on certain dates
        trades_offsetted        = top_flop[top_flop.index.get_level_values('Date').isin(dates_offset[i])]
        trades_offsetted_no_tc  = top_flop_no_tc[top_flop_no_tc.index.get_level_values('Date').isin(dates_offset[i])]
        
        # helper variables so that the mean can be calculated on the go
        helper       = trades_offsetted.resample('D').apply(lambda x : ((1+x).cumprod()-1).iloc[-1])
        helper_no_tc = trades_offsetted_no_tc.resample('D').apply(lambda x : ((1+x).cumprod()-1).iloc[-1])
        
    #    daily_rets_list.append(helper)
    #    daily_rets_no_tc_list.append(helper_no_tc)
        # to build averages over multiple portfolios, take the mean of the generated daily returns
        daily_rets_tc += helper / (120 / n)
        #daily_rets += helper / len(cherry_indices)
        #print(helper.describe())
        daily_rets += helper_no_tc / (120 / n)
        #daily_rets_no_tc += helper_no_tc / len(cherry_indices)
        #print(helper_no_tc.describe())
           
        #helper.cumsum().plot(figsize=(12, 7), title='Offest: ' + str(i))
        #helper_no_tc.cumsum().plot(figsize=(12, 7), title='TC Offest: ' + str(i))
        #plt.show()
        all_means.append(helper.mean())
        all_means_no_tc.append(helper_no_tc.mean())
    
    kpi = functions.get_kpi_array(daily_rets['return_total'])
    kpi_tc = functions.get_kpi_array(daily_rets_tc['return_total'])
    
    
#    daily_rets.to_csv(path_result + model_name + "_" + str(staleness_filter) + ".csv")
#    daily_rets_tc.to_csv(path_result + model_name + "_" + str(staleness_filter) + "_tc.csv")
    
    kpi_df = pd.DataFrame()
    kpi_df[model_name] = kpi
    kpi_df[model_name + "_tc"] = kpi_tc
    kpi_df.index = ['Mean return', 'Standard error', 't-Statistic', 'Minimum', '25% Quantile', 'Median', '75% Quantile', 'Maximum', 'Share > 0',
                    'Standard dev.', 'Skewness', 'Kurtosis']
    
    return kpi_df.round(decimals=6), daily_rets, daily_rets_tc



def get_kpis(offset=10):
    kpis = []
    for n in np.arange(0,120,offset):
        trades = top_flop[top_flop.index.get_level_values('Date').isin(dates_offset[n])]
        kpis.append(functions.get_kpi_array(trades.resample('D').apply(lambda x : ((1+x).cumprod()-1).iloc[-1])['return_total']))

    return kpis

def get_trades():
    trades = []
    for n in np.arange(0,120,10):
        trades_long = top_k_with_vol[top_k_with_vol.index.get_level_values('Date').isin(dates_offset[n])]
        trades_short = flop_k_with_vol[flop_k_with_vol.index.get_level_values('Date').isin(dates_offset[n])]
        trades.append((trades_long, trades_short))
    
    return trades

all_means = pd.concat(all_means)
# average results (total daily returns)
print(all_means.loc['return_total'].describe())

all_means_no_tc = pd.concat(all_means_no_tc)
# average results before transaction costs (total daily returns)
print(all_means_no_tc.loc['return_total'].describe())



def compare_trades(dict_list, top_hack, flop_hack, dates_offsets):
    missing_trades = []
    
    offsets = np.arange(0,120,10)
    
    for i, n  in enumerate(offsets):
        print("I start run " + str(i) + " now.")
        trades_df       = dict_list[i][2][0]
        trades_df       = trades_df[trades_df['Top/Flop']!=-1]
        trades_long     = trades_df[trades_df['Type of trade']=='long']
        trades_short    = trades_df[trades_df['Type of trade']=='short']
        
        trades_long_hack    = top_hack[top_hack.index.get_level_values('Date').isin(dates_offsets[n])]
        trades_short_hack   = flop_hack[flop_hack.index.get_level_values('Date').isin(dates_offsets[n])]
        
        coins_hack_long = list(trades_long_hack.index.get_level_values('Stock'))
        coins_hack_long = [a[:a.find("_")] for a in coins_hack_long]
        trades_long_hack['Coin'] = coins_hack_long
        
        coins_hack_short = list(trades_short_hack.index.get_level_values('Stock'))
        coins_hack_short = [a[:a.find("_")] for a in coins_hack_short]
        trades_short_hack['Coin'] = coins_hack_short

        missing_long    = pd.DataFrame()
        missing_short   = pd.DataFrame()
        extra_long      = pd.DataFrame()
        extra_short     = pd.DataFrame()
        
        trades_long_hack['opening_vol']     = np.round(trades_long_hack['opening_vol'], decimals=1)
        trades_short_hack['opening_vol']    = np.round(trades_short_hack['opening_vol'], decimals=1)
        
        trades_long['Volume entry']     = np.round(trades_long['Volume entry'], decimals=1)
        trades_short['Volume entry']    = np.round(trades_short['Volume entry'], decimals=1)
        
        trades_long_hack['prediction'] = np.round(trades_long_hack['prediction'], decimals=6)
        trades_short_hack['prediction'] = np.round(trades_short_hack['prediction'], decimals=6)
        
        trades_long['Prediction']     = np.round(trades_long['Prediction'], decimals=6)
        trades_short['Prediction']    = np.round(trades_short['Prediction'], decimals=6)
        
#        for j in range(len(trades_long)):
#            helper = trades_long_hack[trades_long_hack['Coin']==trades_long.iloc[j,1]]
#            helper = helper[helper['opening_vol']==trades_long.iloc[j,8]]
#            helper = helper[helper['prediction']==float(trades_long.iloc[j,4])]
#            
#            if len(helper)>1:
#                print("Multiple trades match the criteria in run " + str(j))
#            elif len(helper)==1:
#                continue
#            else:
#                missing_long.append(trades_long.iloc[j,:])
        
#        for j in range(len(trades_short)):            
#            helper = trades_short_hack.loc[trades_short_hack['Coin']==trades_short.iloc[j,1]]
#            helper = helper.loc[helper['opening_vol']==trades_short.iloc[j,8]]
#            helper = helper[helper['prediction']==float(trades_short.iloc[j,4])]
#            
#            if len(helper)>1:
#                print("Multiple trades match the criteria in run " + str(j))
#            elif len(helper)==1:
#                continue
#            else:
#                missing_short.append(trades_short.iloc[j,:])
        
        for coin in sorted(list(set(trades_long['Coin']))):
            helper_long     = trades_long_hack[trades_long_hack['Coin']==coin]
            helper_short    = trades_short_hack[trades_short_hack['Coin']==coin]
            trades_long_helper  = trades_long[trades_long['Coin']==coin]
            trades_short_helper = trades_short[trades_short['Coin']==coin]
            
            helper_long     = helper_long[~helper_long['opening_vol'].isin(trades_long[trades_long['Coin']==coin]['Volume entry'])]
            helper_short    = helper_short[~helper_short['opening_vol'].isin(trades_short[trades_short['Coin']==coin]['Volume entry'])]
#            helper_long     = trades_long[~trades_long['Volume entry'].isin(trades_long_hack[trades_long_hack['Coin']==coin]['opening_vol'])]
#            helper_short    = trades_short[~trades_short['Volume entry'].isin(trades_short_hack[trades_short_hack['Coin']==coin]['opening_vol'])]
            trades_extra_long = trades_long_helper[~trades_long_helper['Volume entry'].isin(helper_long['opening_vol'])]
            trades_extra_short = trades_short_helper[~trades_short_helper['Volume entry'].isin(helper_short['opening_vol'])]
            
            missing_long    = missing_long.append(helper_long)
            missing_short   = missing_short.append(helper_short)
            extra_long      = extra_long.append(trades_extra_long)
            extra_short     = extra_short.append(trades_extra_short)
            
        missing_trades.append((missing_long, extra_long, missing_short, extra_short))
    
    return missing_trades
    

def get_trades_alex():
    trades = []
    for kpi in kpi_twelve:
        trades_helper = kpi[2][0][kpi[2][0]['Top/Flop']!=-1]
        long = trades_helper[trades_helper['Type of trade']=='long']
        short = trades_helper[trades_helper['Type of trade']=='short']
        
        trades.append((long.sort_values(by=['Time entry', 'Coin']), short.sort_values(by=['Time entry', 'Coin'])))
        
    return trades
        
for tuples in trades_alex:
    for df in tuples:
        df.index = [pd.to_datetime(df['Time entry'] + timedelta(minutes=117)), df.Coin]


for tuples in trades_hack:
    for df in tuples:
        df['Time'] = df.index.get_level_values('Date')
        df['Coin'] = df.index.get_level_values('Stock')
        df['Coin'] = [a[:a.find("_")] for a in list(df.Coin)]
        df.index = [pd.to_datetime(df.Time), df.Coin]


joined_dfs = []

for tuples_alex, tuples_hack in zip(trades_alex, trades_hack):
    for i, x in enumerate(zip(tuples_alex, tuples_hack)):
        if i == 0:
            df_alex = pd.DataFrame(x[0]['Return'].groupby(level='Time entry').sum())
            df_hack = pd.DataFrame(x[1]['return_long'].groupby(level='Time').sum())
            joined_dfs.append(df_alex.join(df_hack))
        elif i == 1:
            df_alex = pd.DataFrame(x[0]['Return'].groupby(level='Time entry').sum())
            df_hack = pd.DataFrame(x[1]['return_short'].groupby(level='Time').sum())
            joined_dfs.append(df_alex.join(df_hack))

indices_list = []
for i, entry in enumerate(joined_dfs):
    helper_list = []
    if i % 2 == 0:
        helper_list = np.round(entry['Return'], decimals=6) == np.round(entry['return_long'], decimals=6)
        helper_list = helper_list[helper_list!=1]
    if i % 2 == 1:
        helper_list = np.round(entry['Return'], decimals=6) == np.round(entry['return_short'], decimals=6)
        helper_list = helper_list[helper_list!=1]  
    indices_list.append(helper_list)
    
path = 'C:/Users/adein/Desktop/DailyRets/Paper/final/'    
files = [f for f in listdir(path) if isfile(join(path, f))]
data  = [pd.read_csv(path +f) for f in files]
data_plot = data[:4]

for d in data_plot_tc:
    d.index = pd.to_datetime(d.Date)
    del(d['Date'])
    plt.plot(d['return_total'].cumsum()+1)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(['BR', 'RF_ind', 'LR', 'RF_univ'], fontsize=20)   
plt.title("Cummulative returns w/o TC", fontsize=20) 


def build_complete_df(path_data):
    from os import listdir
    from os.path import isfile, join
    
    data_files = [f for f in listdir(path_data) if isfile(join(path_data, f))]
    
    result_df = pd.DataFrame()
    
    print("[INFO] The data is being read in.")
    for data in data_files:
        coin = data[:data.find("_")]
        helper_df = pd.read_csv(path_data + data)
        helper_df.columns = ['Date', 'Stock', 'prediction']
        
        if coin in ['ADA', 'XVG']:
            helper_df.Stock = coin + "_BitTrex"
        elif coin in ['BCN', 'CND', 'CVC', 'STRAT', 'TNT', 'VIB', 'XDN']:
            helper_df.Stock = coin + "_HitBTC"
        elif coin in ['NXT']:
            helper_df.Stock = coin + "_Poloniex"
        elif coin in ['RDD', 'WAVES', 'XEM']:
            helper_df.Stock = coin + "_Yobit"
        elif coin in ['USDT']:
            helper_df.Stock = coin + "_Kraken"
        else:
            helper_df.Stock = coin + "_Bitfinex"
            
        helper_df.index = [pd.to_datetime(helper_df.Date), helper_df.Stock]
        helper_df.index.names = ['Date', 'Stock']
        result_df = pd.concat([helper_df, result_df], axis=0)
    
    del(result_df['Date'])
    del(result_df['Stock'])
    print("[INFO] The dataframe has been created!")
    return result_df.sort_index(level='Date')


def build_kpis_trade(path_data, raw_data, model_name, model_type='universal', staleness_filter=0.25, tc=0.002):
    print("[INFO] The KPI for a staleness filter of " + str(staleness_filter) + " and TC of " + str(tc) + " are being caluclated!")
    print(model_name)
    if model_type == 'universal':
        predictions = pd.read_csv(path_data)
        predictions.columns = ['Date', 'Stock', 'prediction']
        predictions.index = [pd.to_datetime(predictions.Date), predictions.Stock]
        del(predictions['Date'])
        del(predictions['Stock'])
    else:
        predictions = build_complete_df(path_data)
        
        
    FORECAST_HORIZON = 120
    opening_date_vol = raw_data.volume_from.unstack()
    opening_date_vol = opening_date_vol.shift(FORECAST_HORIZON-1)
    opening_date_vol_stacked = opening_date_vol.stack()
    opening_date_vol_stacked.name = 'opening_vol'
    
    
    # Compute the actual returns (we open at the open and close at the close of the respective bars)
    open_prices = raw_data.open.unstack()
    close_prices = raw_data.close.unstack()
    close_volume = raw_data.volume_from.unstack()
    
    # when no volume for closing is available, bfill with the first row for which volume is available
    for a_col in close_prices.columns:
        close_prices.loc[close_volume[a_col]<=1, a_col] = np.NAN
    close_prices = close_prices.bfill()
    
    actual_returns = close_prices/open_prices.shift(FORECAST_HORIZON-1) - 1
    actual_returns = actual_returns.stack()
    actual_returns.name = 'actual_returns'
    
    
    """ First join the returns to remove those timeseries that do not trade in the trading period """
    predictions = predictions.join(actual_returns)
    predictions = predictions.dropna()
    
    
    K = 5
    top_k = predictions.groupby(level='Date').prediction.nlargest(K)
    top_k.index = top_k.index.droplevel(0)
    top_k = pd.DataFrame(top_k)
    top_k = top_k.join(actual_returns)
    top_k.columns=['prediction', 'return_long']
    
    if model_type == 'universal':
        top_k = top_k[top_k['prediction']>0.5]
    else:
        top_k = top_k[top_k['prediction']>0]
    
    
    flop_k = predictions.groupby(level='Date').prediction.nsmallest(K)
    flop_k.index = flop_k.index.droplevel(0)
    flop_k = pd.DataFrame(flop_k)
    flop_k = flop_k.join(actual_returns)
    flop_k.columns=['prediction', 'return_short']
    flop_k.return_short *= -1
    
    if model_type == 'universal':
        flop_k = flop_k[flop_k['prediction']<0.5]
    else:
        flop_k = flop_k[flop_k['prediction']<0]
    
    
    # Compute the staleness/volume filter
    STALENESS = staleness_filter
    staleness_filter = opening_date_vol.shift(1) # we only use information up to one minute prior to opening the position
    staleness_filter = ((staleness_filter>=1)*1.0).rolling(window=1440).sum()/(1440)
    staleness_filter = staleness_filter.stack()
    staleness_filter = staleness_filter.loc[staleness_filter>=STALENESS]
    staleness_filter.name = 'staleness_filter'
    
    # Apply the staleness filter (we join and drop the items that do not make it through the filter)
    """ (Change Alex) Drop those of the top / flop 5 that do not make it through the filter """
    top_k = top_k.join(staleness_filter)
    top_k = top_k.dropna()
    del(top_k['staleness_filter'])
    
    flop_k = flop_k.join(staleness_filter)
    flop_k = flop_k.dropna()
    del(flop_k['staleness_filter'])
    
    # Average per trade returns just with staneless/volume filter
#    print('[INFO] With staleness filter of', STALENESS)
#    print(top_k.describe())
#    print(flop_k.describe())
    
    
    # Remove trades that have no volume in the opening minute
    top_k_with_vol = top_k.join(opening_date_vol_stacked)
    top_k_with_vol = top_k_with_vol.loc[top_k_with_vol.opening_vol > 1]
    flop_k_with_vol = flop_k.join(opening_date_vol_stacked)
    flop_k_with_vol = flop_k_with_vol.loc[flop_k_with_vol.opening_vol > 1]
    
#    print('[INFO] With staleness filter of', STALENESS, 'and opening volume and exit when volume is available')
#    print(top_k_with_vol['return_long'].describe())
#    print(flop_k_with_vol['return_short'].describe())
    
    
    # Subtract transaction costs and aggregate results
    """ Aggregate to daily returns """
    tops = top_k_with_vol.copy()
    tops['return_long'] -= tc
    
    flops = flop_k_with_vol.copy()
    flops['return_short'] -= tc
    
    kpi_top     = functions.get_kpi_array(tops['return_long'])
    kpi_top     = [len(tops)] + kpi_top
    kpi_flop    = functions.get_kpi_array(flops['return_short'])
    kpi_flop    = [len(flops)] + kpi_flop
    
    kpi_total_helper = pd.DataFrame()
    kpi_total_helper = pd.concat([kpi_total_helper, tops['return_long']])
    kpi_total_helper = pd.concat([kpi_total_helper, flops['return_short']])
    
    kpi_total_helper = kpi_total_helper.sort_index(0, level='Date')
    kpi_total_helper.columns = ['return']
    
    kpi_total = functions.get_kpi_array(kpi_total_helper['return'])
    kpi_total = [len(kpi_total_helper)] + kpi_total
    
    
    kpi_df = pd.DataFrame()
    kpi_df['KPI all trades']    = kpi_total
    kpi_df['KPI long trades']   = kpi_top
    kpi_df['KPI short trades']  = kpi_flop
    
    kpi_df.index = ['No. trades', 'Mean return', 'Standard error', 't-Statistic', 'Minimum', '25% Quantile', 'Median', '75% Quantile', 'Maximum', 'Share > 0',
                    'Standard dev.', 'Skewness', 'Kurtosis']

    return kpi_df.round(decimals=6)


#kpi_br_staleness = [kpi_backtest('C:/Users/adein/Uni/WiMa/Master/Masterarbeit/Paper/Bayesian regression/', raw_data, model_type='individual', staleness_filter=0.05*k, model_name="Bayessian_Regression", tc=0.003) for k in range(1)]
#kpi_rf_ind_staleness = [kpi_backtest('C:/Users/adein/Uni/WiMa/Master/Masterarbeit/Paper/Individual Random Forest/', raw_data, model_type='individual', staleness_filter=0.05*k, model_name="Individual Random Forest", tc=0.003) for k in range(1)]
#kpi_lr_staleness_0_25_tc_25 = [kpi_backtest('C:/Users/adein/Anaconda3/envs/py35/ml_testbench_crypto/log_reg_predictions.csv', raw_data, staleness_filter=0.05*k, model_name="Logistic Regression", tc=0.0025) for k in range(6)]
#kpi_rf_univ_staleness_0_25_tc_30 = [kpi_backtest('C:/Users/adein/Anaconda3/envs/py35/ml_testbench_crypto/predictions_1000_trees_run2.csv', raw_data, staleness_filter=0.05*k, model_name="Universal Random Forest", tc=0.003) for k in range(6)]
