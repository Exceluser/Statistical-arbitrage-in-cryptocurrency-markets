# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 16:46:40 2018

@author: adein
"""

import numpy as np
import pandas as pd
import dataprovider as dp
import scipy.stats as ss
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



def kpi_backtest(path_predictions, raw_data, staleness_filter, model_name, 
                 model_type='universal', modell='classification', tc=0.002):
    """
        Args:
            path_predictions:   absolute path to the csv files with the predictions
            raw_data:           dataframe with the raw data
            staleness_filter:   integer indicating the length of the staleness filter
            model_name:         string representing the name of the model (RF, LR, KNN, ...)
            model_type:         string representing the type of model (universal or one-for-all)
            modell:             string representing the type of predictions (classification or regression)
            tc:                 float representing the transaction costs (in bps)
        Returns:
            kpi_df:             dataframe with the overall kpis
            kpi_trades_df:      dataframe with the kpis on trade level
            daily_rets:         dataframe with the daily returns before transaction costs
            daily_rets_tc:      dataframe with the daily returns after transaction costs
    """
        
    print("[INFO] The KPI for a staleness filter of " + str(staleness_filter) + " and TC of " + str(tc) + " are being caluclated!")
    print(model_name)
    if model_type == 'universal':
        predictions = pd.read_csv(path_predictions)
        predictions.columns = ['Date', 'Stock', 'prediction']
        predictions.index = [pd.to_datetime(predictions.Date), predictions.Stock]
        del(predictions['Date'])
        del(predictions['Stock'])
    else:
        predictions = build_complete_df(path_predictions)
    
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
    """ Drop those of the top / flop 5 that do not make it through the filter """
    top_k = top_k.join(staleness_filter)
    top_k = top_k.dropna()
    del(top_k['staleness_filter'])
    
    flop_k = flop_k.join(staleness_filter)
    flop_k = flop_k.dropna()
    del(flop_k['staleness_filter'])
    
    
    # Remove trades that have no volume in the opening minute
    top_k_with_vol = top_k.join(opening_date_vol_stacked)
    top_k_with_vol = top_k_with_vol.loc[top_k_with_vol.opening_vol > 1]
    flop_k_with_vol = flop_k.join(opening_date_vol_stacked)
    flop_k_with_vol = flop_k_with_vol.loc[flop_k_with_vol.opening_vol > 1]
    
    
    
    # Get trade-level kpis
    tops_trade                  = top_k_with_vol.copy()
    tops_trade['return_long']   -= tc
    flops_trade                 = flop_k_with_vol.copy()
    flops_trade['return_short'] -= tc
    
    
    kpi_top_trades     = get_kpi_array(tops_trade['return_long'])
    kpi_top_trades     = [len(tops_trade)] + kpi_top_trades
    kpi_flop_trades    = get_kpi_array(flops_trade['return_short'])
    kpi_flop_trades    = [len(flops_trade)] + kpi_flop_trades
    
    kpi_trades_total_helper = pd.DataFrame()
    kpi_trades_total_helper = pd.concat([kpi_trades_total_helper, tops_trade['return_long']])
    kpi_trades_total_helper = pd.concat([kpi_trades_total_helper, flops_trade['return_short']])
    
    kpi_trades_total_helper = kpi_trades_total_helper.sort_index(0, level='Date')
    kpi_trades_total_helper.columns = ['return']
    
    kpi_trades_total = get_kpi_array(kpi_trades_total_helper['return'])
    kpi_trades_total = [len(kpi_trades_total_helper)] + kpi_trades_total
    
    
    kpi_trades_df = pd.DataFrame()
    kpi_trades_df['KPI all trades']    = kpi_trades_total
    kpi_trades_df['KPI long trades']   = kpi_top_trades
    kpi_trades_df['KPI short trades']  = kpi_flop_trades
    
    kpi_trades_df.index = ['No. trades', 'Mean return', 'Standard error', 't-Statistic', 'Minimum', '25% Quantile', 'Median', '75% Quantile', 'Maximum', 'Share > 0',
                    'Standard dev.', 'Skewness', 'Kurtosis']
    
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
    

    n = 1
    counter = 0
    no_runs = 120
    for i in np.arange(0,no_runs,n):

        # only get those trades that fit in the offset scheme ---> trades that were executed on certain dates
        trades_offsetted        = top_flop[top_flop.index.get_level_values('Date').isin(dates_offset[i])] 
        trades_offsetted_no_tc  = top_flop_no_tc[top_flop_no_tc.index.get_level_values('Date').isin(dates_offset[i])] 
        
        # helper variables so that the mean can be calculated on the go
        # exception to catch certain cases that appear when there are days without trades
        try:
            helper      = trades_offsetted_no_tc.resample('D').apply(lambda x : ((1+x).cumprod()-1).iloc[-1])
            helper_tc   = trades_offsetted.resample('D').apply(lambda x : ((1+x).cumprod()-1).iloc[-1])            
        except IndexError:
            # counter to know how many exceptions there have been
            counter +=1
        
        # to build averages over multiple portfolios, take the mean of the generated daily returns
        daily_rets += helper / (no_runs / n)
        daily_rets_tc += helper_tc / (no_runs / n)
        
        
        #print(helper.describe())
        #print(helper_tc.describe())
           
        #helper.cumsum().plot(figsize=(12, 7), title='Offest: ' + str(i))
        #helper_no_tc.cumsum().plot(figsize=(12, 7), title='TC Offest: ' + str(i))
        #plt.show()
        all_means.append(helper.mean())
        all_means_no_tc.append(helper_tc.mean())
    print(counter)
    
    # calc the kpi
    kpi = get_kpi_array(daily_rets['return_total'])
    kpi_tc = get_kpi_array(daily_rets_tc['return_total'])
    
    
    kpi_df = pd.DataFrame()
    kpi_df[model_name] = kpi
    kpi_df[model_name + "_tc"] = kpi_tc
    kpi_df.index = ['Mean return', 'Standard error', 't-Statistic', 'Minimum', '25% Quantile', 'Median', '75% Quantile', 'Maximum', 'Share > 0',
                    'Standard dev.', 'Skewness', 'Kurtosis']
    
    return kpi_df.round(decimals=6), kpi_trades_df.round(decimals=6), daily_rets, daily_rets_tc


def build_complete_df(path_data):
    """
        Args: 
            path_data:          absolute path to the data files
        Returns: 
            result_df:          dataframe with the complete timeseries with all 40 coins
    """
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

def get_kpi_array(data):
    """
        Args:
            data:       array or list with the numerical data to be analyzed
        Returns: 
            kpi_list:   a list with the computed kpis
    """

    kpi_list = []
    kpi_list.append(np.mean(data))
    kpi_list.append(ss.sem(data))
    kpi_list.append(ss.ttest_ind(data, np.zeros((len(data),)), equal_var=False)[0])
    kpi_list.append(min(data))  
    kpi_list.append(np.percentile(data, q=25))
    kpi_list.append(np.median(data))
    kpi_list.append(np.percentile(data, q=75))
    kpi_list.append(max(data))  
    helper = [x for x in data if x != 0]
    kpi_list.append(len([x for x in helper if x>0]) / len(helper))
    kpi_list.append(np.std(data))
    kpi_list.append(ss.skew(data))  
    kpi_list.append(ss.kurtosis(data)) 
    
#    kpi_df = pd.DataFrame(kpi_list)
#    kpi_df = kpi_df.transpose()
#    
#    kpi_df.columns = ['Mean Return', 'Standard Error', 't-Statistic', 'Minimum', '25% Quantile',
#                      'Median', '75% Quantile', 'Maximum', 'Share > 0', 'Standard Deviation', 'Skewness',
#                      'Kurtosis']
    
    return kpi_list


