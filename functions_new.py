"""
Created on Sat Feb 17 16:00:57 2018

@author: adein
"""
import requests
#import h5py
import pandas as pd
import re
import time
from datetime import datetime
from datetime import timedelta
import calendar
import numpy as np
import scipy.stats as ss
import os.path
import random
import itertools
import code_paper_modified_robustness as code
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

url_1           = 'https://min-api.cryptocompare.com/data/histominute?fsym='
url_2           = '&tsym=USD&limit=1439&aggregate=1&e='
# The per volume in the last 30 days top 100 traded cryptocurrencies according to coinmarketcap.com 
# as of 25.12.2017 - 18:00
coins_all       = ['BTC', 'ETH', 'BCH', 'USDT', 'LTC', 'XRP', 'ETC', 'IOT', 'QTUM', 'EOS',
                   'DASH', 'BTG', 'XMR', 'ZEC', 'NEO', 'XVG', 'XLM', 'NXT', 'ADA', 'TRX', 
                   'OMG', 'INK', 'HSR', 'XEM', 'EMC2', 'LSK', 'WAVES', 'STRAT', 'POWR', 'BTS', 
                   'DGB', 'BNB', 'DOGE', 'NULS', 'SNT', 'MANA', 'MONA', 'SC',
                   'RDD', 'VET', 'VTC', 'BCC', 'YOYOW', 'SAN', 'BITCNY', 'SALT', 'QASH', 'WTC', 
                   'ICX', 'RCN', 'QSP', 'REP', 'VOX', 'PAY', 'ARDR', 'KMD', 'MTL', 'ARK', 
                   'KNC', 'FCT', 'CVC', 'VIB', 'ADX', 'RDN', 'XZC', 'ELF', 'REQ', 'HMQ', 
                   'STORJ', 'BAY', 'SYS', 'EDG', 'BAT', 'GNT', 'BCD', 'MEME', 'GXS', 'CTR', 
                   'LINK', 'ETP', 'DATA', 'XDN', 'NAV', 'BCN', 'CND', 'ZRX', 'TNT', 'STEEM', 
                   'RISE', 'TNB', 'ETN', 'GRS', 'ENG','IOP', 'CMT', 'PTOY', 'PIVX', 'GUP']

coins_filt      = ['BTC', 'ETH', 'BCH', 'USDT', 'LTC', 'XRP', 'ETC', 'QTUM', 'EOS',
                   'DASH', 'BTG', 'XMR', 'ZEC', 'NEO', 'XVG', 'XLM', 'NXT', 'ADA', 'TRX', 
                   'OMG', 'HSR', 'XEM', 'EMC2', 'LSK', 'WAVES', 'STRAT', 'POWR', 'BTS', 
                   'DGB', 'BNB', 'DOGE', 'NULS', 'SNT', 'MANA', 
                   'RDD', 'VTC', 'YOYOW', 'SAN', 'SALT', 'QASH', 'WTC', 
                   'ICX', 'RCN', 'QSP', 'VOX', 'PAY', 'ARDR', 'KMD', 'MTL', 'ARK', 
                   'KNC', 'FCT', 'CVC', 'VIB', 'ADX', 'REQ', 'HMQ', 
                   'STORJ', 'BAY', 'EDG', 'BAT', 'GNT', 'MEME', 'GXS', 'CTR', 
                   'LINK', 'ETP', 'DATA', 'XDN', 'NAV', 'BCN', 'CND', 'ZRX', 'TNT', 'STEEM', 
                   'RISE', 'TNB', 'ETN', 'GRS', 'ENG','IOP', 'PTOY', 'GUP']

# All exchanges accessable via the cryptocompare API
exchanges       = ['Cryptsy', 'BTCChina', 'Bitstamp', 'BTER', 'OKCoin', 'Coinbase', 'Poloniex', 
                   'Cexio', 'BTCE', 'BitTrex', 'Kraken', 'Bitfinex', 'Yacuna',  
                   'Yunbi', 'itBit', 'HitBTC', 'btcXchange', 'BTC38', 'Coinfloor', 'Huobi', 
                   'CCCAGG', 'LakeBTC', 'ANXBTC', 'Bit2C', 'Coinsetter', 'CCEX', 'Coinse', 
                   'MonetaGo', 'Gatecoin', 'Gemini', 'CCEDK', 'Cryptopia', 'Exmo', 'Yobit', 'Korbit', 
                   'BitBay', 'BTCMarkets', 'Coincheck', 'QuadrigaCX', 'BitSquare', 'Vaultoro', 
                   'MercadoBitcoin', 'Bitso', 'Unocoin', 'BTCXIndia', 'Paymium', 'TheRockTrading', 
                   'bitFlyer', 'Quoine', 'Luno', 'EtherDelta', 'bitFlyerFX', 'TuxExchange', 'CryptoX', 
                   'Liqui', 'MtGox', 'BitMarket', 'LiveCoin', 'Coinone', 'Tidex', 'Bleutrade', 
                   'EthexIndia', 'Bithumb', 'CHBTC', 'ViaBTC', 'Jubi', 'Zaif', 'Novaexchange', 
                   'WavesDEX', 'Binance', 'Lykke', 'Remitano', 'Coinroom', 'Abucoins', 'BXinth', 
                   'Gateio', 'HuobiPro', 'OKEX']

exchanges_west  = ['Bitstamp', 'Coinbase', 'Poloniex', 'Cexio', 'BitTrex', 'Kraken', 'Bitfinex', 'itBit',
                   'Coinfloor','CCCAGG', 'ANXBTC', 'Coinsetter', 'CCEX', 'MonetaGo', 'Gemini', 'CCEDK', 
                   'Cryptopia', 'Exmo', 'BitBay', 'QuadrigaCX', 'Vaultoro', 'Paymium', 'TheRockTrading',
                   'Luno', 'TuxExchange', 'LiveCoin', 'Tidex', 'Novaexchange', 'Lykke', 'Abucoins',
                   'Gateio'] 

filt_ex_liquid  = ['Bitfinex', 'Bitfinex', 'Bitfinex', 'Kraken', 'Bitfinex', 'Bitfinex', 'Bitfinex', 'Bitfinex', 'Bitfinex',
                   'Bitfinex', 'Bitfinex', 'CCCAGG', 'CCCAGG', 'Bitfinex', 'CCCAGG', 'Poloniex', 'CCCAGG', 'CCCAGG', 'CCCAGG', 
                   'Bitfinex', 'CCCAGG', 'CCCAGG', 'CCCAGG', 'CCCAGG', 'Yobit', 'CCCAGG', 'CCCAGG', 'CCCAGG', 
                   'CCCAGG', 'CCCAGG', 'CCCAGG', 'CCCAGG', 'CCCAGG', 'CCCAGG',  
                   'CCCAGG', 'CCCAGG', 'Bitfinex', 'CCCAGG', 'CCCAGG', 'Bitfinex', 'CCCAGG', 
                   'CCCAGG', 'CCCAGG', 'CCCAGG', 'CCCAGG', 'CCCAGG', 'CCCAGG', 'CCCAGG', 'CCCAGG', 'CCCAGG', 
                   'CCCAGG', 'CCCAGG', 'CCCAGG', 'CCCAGG', 'CCCAGG', 'CCCAGG', 'CCCAGG', 
                   'CCCAGG', 'CCCAGG', 'CCCAGG', 'CCCAGG', 'CCCAGG', 'CCCAGG', 'CCCAGG', 'CCCAGG', 
                   'CCCAGG', 'Bitfinex', 'Bitfinex', 'CCCAGG', 'CCCAGG', 'CCCAGG', 'CCCAGG', 'CCCAGG', 'CCCAGG', 'CCCAGG', 
                   'CCCAGG', 'CCCAGG', 'CCCAGG', 'CCCAGG', 'CCCAGG', 'CCCAGG', 'CCCAGG', 'CCCAGG']


def filterExchanges_single(exchanges, currency):
    starter = time.time()
    url_list            = ['https://min-api.cryptocompare.com/data/histominute?fsym=' + currency 
                           + '&tsym=USD&limit=1439&aggregate=1&e=' + exchanges[i] for i in range(len(exchanges))]
    data_list           = [requests.get(url_list[i]).json()['Data'] for i in range(len(url_list))]
    indices             = [i for i in range(len(data_list)) if data_list[i] != []]
    exchanges_cleaned   = [exchanges[i] for i in indices]
    print(time.time() - starter)
    return exchanges_cleaned


def filterExchanges_all(exchanges, currencies):
    exchanges_currencies = [filterExchanges_single(exchanges, currencies[i]) for i in range(len(currencies))]
    return exchanges_currencies


def getTimestamps():
    now         = int(time.time())
    second      = datetime.datetime.today().second
    minute      = datetime.datetime.today().minute
    hour        = datetime.datetime.today().hour
    midnight    = now - second - minute*60 - hour*3600
    
    timestamps = []
    # as only 7 days from now back are possible, 6 makes more sense here
    for i in range(6):
        append_me = str(midnight - i*24*3600)
        timestamps.append(append_me)
    timestamps.reverse()
    
    return timestamps


def getFilteredURLs(filtered_exchanges, filtered_coins, x):
    #start = time.time()
    filtered_urls = [[] for i in range(len(filtered_exchanges))]

    for i in range(len(filtered_coins)):
        for j in range(len(filtered_exchanges[i])):
            filtered_urls[i].append(url_1 + filtered_coins[i] + url_2 
                                     + filtered_exchanges[i][j] + '&toTs=' + str(getTimestamps()[x]))
    
    #print(time.time() - start)
    return filtered_urls


def getURLsTimestamps(exchanges, coins):
    urls = [''] * 6
    for i in range(len(urls)):
        urls[i] = getFilteredURLs(exchanges, coins, i)
    
    return urls


def getData(url_list):
    #start           = time.time()
    data_list       = [[] for i in range(len(url_list))]
    exchange_list   = [[] for i in range(len(url_list))]
    
    for i in range(len(data_list)):
        for j in range(len(url_list[i])):
            if (requests.get(url_list[i][j]).json()['Data'] != []):
                data_list[i].append(requests.get(url_list[i][j]).json()['Data'])
                exchange_index_start    = url_list[i][j].find('&e=') + 3
                exchange_index_end      = url_list[i][j].find('&toTs') 
                exchange_list[i].append(url_list[i][j][exchange_index_start:exchange_index_end])
            else:
                continue
    
    #print("It took %f seconds to complete this action!" %(time.time()-start))
    return data_list, exchange_list


def getAllData(url_list):
    start           = time.time()
    data_list       = [''] * len(url_list)
    exchange_list   = [''] * len(url_list)
    
    for i in range(len(data_list)):
        data_list[i], exchange_list[i] = getData(url_list[i])
        
        
    print("It took %f seconds to get the data for all %i timestamps" %(time.time() - start, len(data_list)))
    return data_list, exchange_list


def createHDFDataframe(data, exchanges, coins):
    hdf_dataframe       = pd.DataFrame()
    date_helperfunc     = np.vectorize(datetime.datetime.fromtimestamp)
    
    for i in range(len(data)):
        if (data[i] != []):
            helper              = pd.DataFrame(data[i])
            helper['Exchange']  = exchanges[i]
            helper['Coin']      = coins
            helper['Datetime']  = date_helperfunc(helper['time']).tolist()
            hdf_dataframe       = hdf_dataframe.append(helper)
        else:
            continue
        
    return hdf_dataframe


def createBigHDFDataframe(data, exchanges, coins):
    #start           = time.time()
    hdf_dataframe   = pd.DataFrame()
    
    assert (len(data) == len(exchanges) & len(exchanges) == len(coins)), "The inputs are not of consistent format"
    
    for i in range(len(data)):
        helper          = createHDFDataframe(data[i], exchanges[i], coins[i])
        hdf_dataframe   = hdf_dataframe.append(helper)
    
    hdf_dataframe['Coin']       = hdf_dataframe.Coin.astype('category')
    hdf_dataframe['Exchange']   = hdf_dataframe.Exchange.astype('category')
    
    #print("It took %f seconds to create this big Dataframe!" %(time.time()-start))
    return hdf_dataframe


# As the consecutive days are needed for the creation of the timeseries, this function helps to get the
# the following day
def timeseries_date_finder(date, day, month):
    assert (day < 32 and month < 13),"The entered day or month are not valid!"
    
    today       = datetime.datetime.today().day
    this_month  = datetime.datetime.today().month    
    
    # Check if still in same month
    if (day < calendar.monthrange(2018, int(date[date.find('_')+1:len(date)]))[1] and month < this_month):
        day = day + 1
        print("A")
    # Check if final day of month
    elif (day == calendar.monthrange(2018, int(date[date.find('_')+1:len(date)]))[1] and month < this_month):
        day = 1
        month = month + 1
        print("B")
    # Check if in final month
    elif (day < today and month == this_month):
        day = day + 1
        print("C")
    
    return day, month

    

def get_liqui_ex_data(data, coins, exchanges, path):
    """ This function creates per coin a .csv file with the timeseries of the coin on the most liquid exchange
        it is being traded on (or CCCAGG if no very liquid exchange is available)
    Args:
        data: A big dataframe with all the data of all the coins on each exchange, ie. the complete timeseries,
              consolidated on exchange level (multiindex ['Coin', 'Exchange', 'Datetime'])
        coins: A list of the filtered coins
        exchanges: A list of the most liquid exchanges, respective to the coins. Must be ordered according to
                   the coins
        path: A absolute path for a directory in which the resulting .csv files should be saved
    Returns:
        error_list: A list with those (coin, exchange) tuples that did not work
    """    
    error_list = []
    
    for i in range(len(coins)):
        if exchanges[i] != '':
            print(str(coins[i]) + " on exchange " + str(exchanges[i]))
            name = path + str(coins[i]) + "_" + str(exchanges[i]) + ".csv"
            
            # If the file does not exist, try to create it
            if os.path.isfile(name) == False:
                print(coins[i])
                helper = data.loc[coins[i]]
                try:
                    helper = helper.loc[exchanges[i]]
                except KeyError:
                    error_list.append((coins[i], exchanges[i]))
                    
                helper.to_csv(name)
            else:
                continue
                
        else:
            continue
    return error_list
        

def get_hours(df):
    """ This function returns the hour of the day in which one potentially trades (UTC+2, Berlin)
    Args:
        df:         A dataframe of the timeseries of the coin to be evaluated
    Returns:
        time_hours: A list with each entry being a number between 0 and 23, representing in which hour one
                    is at the moment, so to speak
    """
    helper_func = np.vectorize(datetime.datetime.fromtimestamp)
    time        = df['Time']
    time        = time[int(2*len(time)/3):]
    time_dates  = helper_func(time)
    time_hours  = [time_dates[i].hour for i in range(len(time_dates))]  
    return time_hours
    

def get_returns(data, step, trading_period):
    """ This function calculates the returns of a prices timeseries
    Args:
        data:           An array or a list or a series with the prices
        step:           The timeinterval of the returns, e. g. 1 for minutely returns, 60 for hourly etc
        trading_period: A boolean to indicate wether to calculate the returns of all prices or only the last third. 
                        Default is True which means only the last third
    Returns:
        returns_list:   A list with the according returns
    """
    returns_list = []
    
    if trading_period == True:
        timeframe = int(2*len(data)/3)
        for i in range(timeframe, len(data)-step):
            if data[i] != 0:
                returns_list.append(data[i+step] / data[i] - 1)
            else:
                returns_list.append(0)
    else:
        timeframe = len(data)
        for i in range(timeframe-step):
            if data[i] != 0:
                returns_list.append(data[i+step] / data[i] - 1)
            else:
                returns_list.append(0)
    
    return np.array(returns_list)

def find_constants(matrix):
    constants = []
    
    for i in range(len(matrix)):
        helper = matrix[i,:]
        zeros  = np.count_nonzero(helper)
        if zeros < len(helper) / 10:
            constants.append(i)
    
    return constants


def readin_importances(path_importances_file):
    """ This function reads in the feature importances from a .csv file
    Args:
        path_importances_file: The absolute path of the file
    Returns:
        importances: A Dataframe with the feature importances
    """
    importances = pd.read_csv(path_importances_file)
    if len(importances.columns) == 1:
        importances = pd.read_csv(path_importances_file, delimiter=";")
        
    importances = importances[importances.columns[1:]]
    return importances       


def get_avg_importances(path_data):
    """ This function calculates the average importances per feature over several files
    Args:
        path_data: The path to the directory in which the files are located
    Returns:
        avg_df: A dataframe with the average importances per feature over all files given in the location
    """
    files = [f for f in listdir(path_data) if isfile(join(path_data, f))]
    
    files_data      = [readin_importances(path_data + file) for file in files]
    files_values    = [files.values for files in files_data]
    files_values    = [values[:,1:] for values in files_values]

    columns = list(files_data[0].columns)
    features = list(files_data[0]['Feature'])
    

    cum_mat = np.zeros(np.shape(files_values[0]))
    for i in range(len(files_values)):
        cum_mat = cum_mat + files_values[i]
    
    avg_mat = cum_mat / len(files_values)
    
    avg_df = pd.DataFrame(avg_mat)
    avg_df.columns = columns[1:]
    avg_df['Feature'] = features
    avg_df = avg_df[['Feature'] + columns[1:]]
    
    return avg_df    


def create_random_walk(data):
    """ This function creates a random walk based on the data given as input
    Args:
        data: An array or a list with prices
    Returns:
        rw: A random walk based on several stylized facts of the data given
    """
    x       = len(data)
    std     = np.std(get_returns(data,1, False))
    start   = np.mean(data)
    rw      = [start]
    
    for i in range(x-1):
        rw.append(rw[i] + random.normalvariate(0,std*start))
    
    return np.array(rw)


def get_correlations(path_data, step, trading_period=True):
    """ This function returns the correlations of the returns of the data given.
    """
    data_files  = [f for f in listdir(path_data) if isfile(join(path_data, f))]
    files       = [pd.read_csv(path_data + f) for f in data_files]
    coins       = [str(f)[0:str(f).find("_")] for f in data_files] 
    close       = [np.array(f['Close']) for f in files]
    min_len     = min([len(close[i]) for i in range(len(close))])
    close_adj   = [close[i][-min_len:] for i in range(len(close))]
    ret_mat     = np.zeros((len(close),min_len-1))
    for i in range(len(close)):
        ret_mat[i,:] = get_returns(close_adj[i], 1, trading_period)
    
    corr_df = pd.DataFrame(np.corrcoef(ret_mat))
    corr_df.columns = coins
    corr_df['Coin'] = coins
    corr_df = corr_df[['Coin'] + coins]
    return corr_df


def calc_rmse(data, predictions, step):
    """ This function calculates the root mean squared error of the predicted returns
    Args:
        data: a array or list with the prices
        predictions: a array or list or series with the predictions
        step: the step size of the predictions
    Returns:
        rmse: a float representing the rmse
    """
    return_actual = get_returns(data, step, False)
    
    if step > 1:
        return_pred   = np.array(predictions['Step ' + str(step)])[:-(step-1)]
    else:
        return_pred   = np.array(predictions['Step ' + str(step)])
    
    return_actual = np.array(return_actual[-len(return_pred):])
    
    rmse = np.sqrt(mean_squared_error(return_actual, return_pred))
    
    return round(rmse, 4)

def rmse_df(path_data, path_pred, coin='all', steps=[1,10,30,60,120]):
    data_files  = [f for f in listdir(path_data) if isfile(join(path_data, f))]
    pred_files  = [f for f in listdir(path_pred) if isfile(join(path_pred, f))]
    coins       = [str(f)[0:str(f).find("_")] for f in data_files]

    result_mat = np.zeros((len(data_files),len(steps)))
    
    for i in range(len(data_files)):
        data = pd.read_csv(path_data + data_files[i])
        pred = pd.read_csv(path_pred + pred_files[i])
        
        close = data['Close']
        
        rmse_helper_list = [0 for j in range(len(steps))]
        for j in range(len(steps)):
            rmse_helper_list[j] = calc_rmse(close, pred, steps[j])
        
        result_mat[i,:] = rmse_helper_list
    
    result_df = pd.DataFrame(result_mat)
    result_df.columns = [str(i) + ' Min' for i in steps]
    result_df['Coin']  = coins
    result_df = result_df[['Coin'] + [str(i) + ' Min' for i in steps]]
    
    return result_df

def get_kpi_array(data):

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
    
    kpi_df = pd.DataFrame(kpi_list)
    kpi_df = kpi_df.transpose()
    
    kpi_df.columns = ['Mean Return', 'Standard Error', 't-Statistic', 'Minimum', '25% Quantile',
                      'Median', '75% Quantile', 'Maximum', 'Share > 0', 'Standard Deviation', 'Skewness',
                      'Kurtosis']
    
    return kpi_list
    
def crosssection_kpi(path_data, path_pred, df=None, top=10, steps=[1,10,30,60,120], ensemble=False, tc=0.002, 
                     staleness_option='volume', filter_intensity=0.25, return_on_committed_capital=True,
                     return_on_invested_capital=False, fill_k=False, volume_factor=1.0,
                     model_type='Classification', offset=0, closing=0):        
    
    if type(top) != list:
        top = [top]
    
    kpi_mat             = np.zeros((len(steps)*len(top), 13))
    kpi_trade           = np.zeros((len(steps)*len(top), 17))
    kpi_trade_pos       = np.zeros((len(steps)*len(top), 17))
    kpi_trade_neg       = np.zeros((len(steps)*len(top), 17))
    
    kpi_trade_cc           = np.zeros((len(steps)*len(top), 17))
    kpi_trade_pos_cc       = np.zeros((len(steps)*len(top), 17))
    kpi_trade_neg_cc       = np.zeros((len(steps)*len(top), 17))
    
    kpi_mat_tc          = np.zeros((len(steps)*len(top), 13))
    kpi_trade_tc        = np.zeros((len(steps)*len(top), 17))
    kpi_trade_pos_tc    = np.zeros((len(steps)*len(top), 17))
    kpi_trade_neg_tc    = np.zeros((len(steps)*len(top), 17))
    
    kpi_trade_tc_cc        = np.zeros((len(steps)*len(top), 17))
    kpi_trade_pos_tc_cc    = np.zeros((len(steps)*len(top), 17))
    kpi_trade_neg_tc_cc    = np.zeros((len(steps)*len(top), 17))

    holding_period_analysis = np.zeros((len(steps)*len(top), 12))
    holding_period_analysis_cc = np.zeros((len(steps)*len(top), 12))
    
    daily_rets_list     = []
    daily_rets_tc_list  = []
    
    df_list = []
    
    helper = 0
    for step in steps:
        for k in top:
            # get the dataframe with the stats of each trade
            if df is not None:
                df = df
            else:
                df = code.evaluate_performance_crosssection(path_data, path_pred, top=k, time_step=step, ensemble=ensemble, 
                                                           staleness_option=staleness_option, filter_intensity=filter_intensity,
                                                           return_on_committed_capital=return_on_committed_capital,
                                                           return_on_invested_capital=return_on_invested_capital,
                                                           fill_k=fill_k, volume_factor=volume_factor, 
                                                           model_type=model_type, offset=offset, closing=closing)

            df_list.append(df)

            # get kpis on trade level
            holding_period_analysis[helper,:]   = analyze_holding_period(df, timestep=step)
            helper_pos                          = df[df['Return']>0]
            helper_neg                          = df[df['Return']<0]
            helper_long                         = df[df['Type of trade']=='long']
            helper_long_pos                     = helper_long[helper_long['Return']>0]
            helper_long_neg                     = helper_long[helper_long['Return']<0]
            helper_short                        = df[df['Type of trade']=='short']
            helper_short_pos                    = helper_short[helper_short['Return']>0]
            helper_short_neg                    = helper_short[helper_short['Return']<0]
            
            df_cc                                  = df[df['Top/Flop']!=-1]
            holding_period_analysis_cc[helper,:]   = analyze_holding_period(df_cc, timestep=step)
            helper_pos_cc                          = df[(df['Return']>0) & (df['Top/Flop']!=-1)]
            helper_neg_cc                          = df[(df['Return']<0) & (df['Top/Flop']!=-1)]
            helper_long_cc                         = df[(df['Type of trade']=='long') & (df['Top/Flop']!=-1)]
            helper_long_pos_cc                     = helper_long[(helper_long['Return']>0) & (df['Top/Flop']!=-1)]
            helper_long_neg_cc                     = helper_long[(helper_long['Return']<0) & (df['Top/Flop']!=-1)]
            helper_short_cc                        = df[(df['Type of trade']=='short') & (df['Top/Flop']!=-1)]
            helper_short_pos_cc                    = helper_short[(helper_short['Return']>0) & (df['Top/Flop']!=-1)]
            helper_short_neg_cc                    = helper_short[(helper_short['Return']<0) & (df['Top/Flop']!=-1)]
            
            kpi_trade[helper,3]                 = np.mean(np.array(helper_long['Return']))
            kpi_trade[helper,4]                 = np.mean(np.array(helper_short['Return']))
            kpi_trade[helper,5:]                = get_kpi_array(np.array(df['Return']))
            
            kpi_trade_cc[helper,3]                 = np.mean(np.array(helper_long_cc['Return']))
            kpi_trade_cc[helper,4]                 = np.mean(np.array(helper_short_cc['Return']))
            kpi_trade_cc[helper,5:]                = get_kpi_array(np.array(df_cc['Return']))
            
            kpi_trade_pos[helper,1]             = len(helper_long_pos)
            kpi_trade_pos[helper,2]             = len(helper_short_pos)
            kpi_trade_pos[helper,3]             = np.mean(np.array(helper_long_pos['Return']))
            kpi_trade_pos[helper,4]             = np.mean(np.array(helper_short_pos['Return']))
            kpi_trade_pos[helper,5:]            = get_kpi_array(np.array(helper_pos['Return']))
            
            kpi_trade_pos_cc[helper,1]             = len(helper_long_pos_cc)
            kpi_trade_pos_cc[helper,2]             = len(helper_short_pos_cc)
            kpi_trade_pos_cc[helper,3]             = np.mean(np.array(helper_long_pos_cc['Return']))
            kpi_trade_pos_cc[helper,4]             = np.mean(np.array(helper_short_pos_cc['Return']))
            kpi_trade_pos_cc[helper,5:]            = get_kpi_array(np.array(helper_pos_cc['Return']))
            
            kpi_trade_neg[helper,1]             = len(helper_long_neg)
            kpi_trade_neg[helper,2]             = len(helper_short_neg)
            kpi_trade_neg[helper,3]             = np.mean(np.array(helper_long_neg['Return']))
            kpi_trade_neg[helper,4]             = np.mean(np.array(helper_short_neg['Return']))
            kpi_trade_neg[helper,5:]            = get_kpi_array(np.array(helper_neg['Return']))
            
            kpi_trade_neg_cc[helper,1]             = len(helper_long_neg_cc)
            kpi_trade_neg_cc[helper,2]             = len(helper_short_neg_cc)
            kpi_trade_neg_cc[helper,3]             = np.mean(np.array(helper_long_neg_cc['Return']))
            kpi_trade_neg_cc[helper,4]             = np.mean(np.array(helper_short_neg_cc['Return']))
            kpi_trade_neg_cc[helper,5:]            = get_kpi_array(np.array(helper_neg_cc['Return']))
            
            kpi_trade_tc[helper,3]              = np.mean(np.array([helper_long.iloc[i,3]-tc if helper_long.iloc[i,7] != -1 else 0 for i in range(len(helper_long))]))
            kpi_trade_tc[helper,4]              = np.mean(np.array([helper_short.iloc[i,3]-tc if helper_short.iloc[i,7] != -1 else 0 for i in range(len(helper_short))]))
            kpi_trade_tc[helper,5:]             = get_kpi_array(np.array([df.iloc[i,3]-tc if df.iloc[i,7] != -1 else 0 for i in range(len(df))]))            
            
            kpi_trade_tc_cc[helper,3]              = np.mean(np.array(helper_long_cc['Return'])-tc)
            kpi_trade_tc_cc[helper,4]              = np.mean(np.array(helper_short_cc['Return'])-tc)
            kpi_trade_tc_cc[helper,5:]             = get_kpi_array(np.array(df_cc['Return'])-tc)            
            
            kpi_trade_pos_tc[helper,1]          = len(helper_long_pos)
            kpi_trade_pos_tc[helper,2]          = len(helper_short_pos)
            kpi_trade_pos_tc[helper,3]          = np.mean(np.array([helper_long_pos.iloc[i,3]-tc if helper_long_pos.iloc[i,7] != -1 else 0 for i in range(len(helper_long_pos))]))
            kpi_trade_pos_tc[helper,4]          = np.mean(np.array([helper_short_pos.iloc[i,3]-tc if helper_short_pos.iloc[i,7] != -1 else 0 for i in range(len(helper_short_pos))]))
            kpi_trade_pos_tc[helper,5:]         = get_kpi_array(np.array([helper_pos.iloc[i,3]-tc if helper_pos.iloc[i,7] != -1 else 0 for i in range(len(helper_pos))]))      
            
            kpi_trade_pos_tc_cc[helper,1]          = len(helper_long_pos_cc)
            kpi_trade_pos_tc_cc[helper,2]          = len(helper_short_pos_cc)
            kpi_trade_pos_tc_cc[helper,3]          = np.mean(np.array(helper_long_pos_cc['Return'])-tc)
            kpi_trade_pos_tc_cc[helper,4]          = np.mean(np.array(helper_short_pos_cc['Return'])-tc)
            kpi_trade_pos_tc_cc[helper,5:]         = get_kpi_array(np.array(helper_pos_cc['Return'])-tc)
            
            kpi_trade_neg_tc[helper,1]          = len(helper_long_neg)
            kpi_trade_neg_tc[helper,2]          = len(helper_short_neg)
            kpi_trade_neg_tc[helper,3]          = np.mean(np.array([helper_long_neg.iloc[i,3]-tc if helper_long_neg.iloc[i,7] != -1 else 0 for i in range(len(helper_long_neg))]))
            kpi_trade_neg_tc[helper,4]          = np.mean(np.array([helper_short_neg.iloc[i,3]-tc if helper_short_neg.iloc[i,7] != -1 else 0 for i in range(len(helper_short_neg))]))
            kpi_trade_neg_tc[helper,5:]         = get_kpi_array(np.array([helper_neg.iloc[i,3]-tc if helper_neg.iloc[i,7] != -1 else 0 for i in range(len(helper_neg))]))
            
            kpi_trade_neg_tc_cc[helper,1]          = len(helper_long_neg_cc)
            kpi_trade_neg_tc_cc[helper,2]          = len(helper_short_neg_cc)
            kpi_trade_neg_tc_cc[helper,3]          = np.mean(np.array(helper_long_neg_cc['Return'])-tc)
            kpi_trade_neg_tc_cc[helper,4]          = np.mean(np.array(helper_short_neg_cc['Return'])-tc)
            kpi_trade_neg_tc_cc[helper,5:]         = get_kpi_array(np.array(helper_neg_cc['Return'])-tc)
            
            
            ret_tc_helper           = np.array(df[df['Top/Flop'] != -1]['Return'])-tc
            kpi_trade[helper,13]    = round(np.sum([1 if np.array(df[df['Top/Flop'] != -1]['Return'])[i] > 0 else 0 for i in range(len(df[df['Top/Flop'] != -1]))]) / len(df[df['Top/Flop'] != -1]), 2)
            kpi_trade_pos_tc[helper,13] = round(np.sum([1 if np.array(helper_pos['Return'])[i] > tc else 0 for i in range(len(helper_pos))]))
            kpi_trade_neg_tc[helper,13] = round(np.sum([1 if np.array(helper_neg['Return'])[i] > tc else 0 for i in range(len(helper_neg))]))
            kpi_trade_tc[helper,13] = round(np.sum([1 if ret_tc_helper[i] > 0 else 0 for i in range(len(ret_tc_helper))]) / len(ret_tc_helper), 2)
                
            
            # calculate the daily returns
            daily_rets, _, no_days, acc_df, day_end_df      = calc_daily_ret_new(df, tc=0)
            daily_rets_list.append((acc_df, day_end_df))
            
            daily_rets_tc, _, no_days, acc_df_tc, day_end_df_tc   = calc_daily_ret_new(df, tc=tc)
            daily_rets_tc_list.append((acc_df_tc, day_end_df_tc))
            
#            daily_rets, no_days, _, _ , ret_per_day, ret_accum = calc_daily_returns_cross_eva(df, tc=0)
#            daily_rets_list.append((ret_per_day, ret_accum))
#            
#            daily_rets_tc, _, _, _ , ret_per_day_tc, ret_accum_tc = calc_daily_returns_cross_eva(df, tc=tc)
#            daily_rets_tc_list.append((ret_per_day_tc, ret_accum_tc))

            kpi_trade[helper,0] = round(len(df)/no_days,2)
            kpi_trade[helper,1] = len(helper_long)
            kpi_trade[helper,2] = len(helper_short)
            
            kpi_trade_cc[helper,0] = round(len(df_cc)/no_days,2)
            kpi_trade_cc[helper,1] = len(helper_long_cc)
            kpi_trade_cc[helper,2] = len(helper_short_cc)
            
            kpi_trade_pos[helper,0] = round(len(helper_pos) / no_days, 2)
            kpi_trade_neg[helper,0] = round(len(helper_neg) / no_days, 2)
            kpi_trade_pos_tc[helper,0] = round(len(helper_pos[helper_pos['Return']>tc]) / no_days, 2)
            kpi_trade_neg_tc[helper,0] = round(len(helper_neg) / no_days, 2)
            
            kpi_trade_pos_cc[helper,0] = round(len(helper_pos_cc) / no_days, 2)
            kpi_trade_neg_cc[helper,0] = round(len(helper_neg_cc) / no_days, 2)
            kpi_trade_pos_tc_cc[helper,0] = round(len(helper_pos_cc[helper_pos_cc['Return']>tc]) / no_days, 2)
            kpi_trade_neg_tc_cc[helper,0] = round(len(helper_neg_cc) / no_days, 2)
            
            kpi_trade_tc[helper,0] = round(len(df)/no_days,2)
            kpi_trade_tc[helper,1] = len(helper_long)
            kpi_trade_tc[helper,2] = len(helper_short)
            
            kpi_trade_tc_cc[helper,0] = round(len(df_cc)/no_days,2)
            kpi_trade_tc_cc[helper,1] = len(helper_long_cc)
            kpi_trade_tc_cc[helper,2] = len(helper_short_cc)
            
            
            # fill the kpi matrix
            kpi_mat[helper,0]    = len(df)
            kpi_mat_tc[helper,0] = len(df)
            
            kpi_mat[helper,1:]      = get_kpi_array(np.array(daily_rets))
            kpi_mat_tc[helper,1:]   = get_kpi_array(np.array(daily_rets_tc))

            helper += 1
    
    holding_period_analysis_df = pd.DataFrame(holding_period_analysis)
    holding_period_analysis_cc_df = pd.DataFrame(holding_period_analysis_cc)
    kpi_df                  = pd.DataFrame(kpi_mat)
    kpi_trade_df            = pd.DataFrame(kpi_trade)
    kpi_trade_pos_df        = pd.DataFrame(kpi_trade_pos)
    kpi_trade_neg_df        = pd.DataFrame(kpi_trade_neg)
    kpi_trade_cc_df            = pd.DataFrame(kpi_trade_cc)
    kpi_trade_pos_cc_df        = pd.DataFrame(kpi_trade_pos_cc)
    kpi_trade_neg_cc_df        = pd.DataFrame(kpi_trade_neg_cc)
    kpi_tc_df               = pd.DataFrame(kpi_mat_tc)
    kpi_trade_tc_df         = pd.DataFrame(kpi_trade_tc)
    kpi_trade_pos_tc_df     = pd.DataFrame(kpi_trade_pos_tc)
    kpi_trade_neg_tc_df     = pd.DataFrame(kpi_trade_neg_tc)
    kpi_trade_tc_cc_df         = pd.DataFrame(kpi_trade_tc_cc)
    kpi_trade_pos_tc_cc_df     = pd.DataFrame(kpi_trade_pos_tc_cc)
    kpi_trade_neg_tc_cc_df     = pd.DataFrame(kpi_trade_neg_tc_cc)
    
    
    trade_columns       = ['No Trades Daily', 'No Long Trades', 'No Short Trades', 'Mean Return (long)', 'Mean Return (short)', 'Mean Return', 'Standard Error', 't-Statistic', 'Minimum', '25% Quantile',
                               'Median', '75% Quantile', 'Maximum', 'Share > 0', 'Standard Deviation', 'Skewness',
                               'Kurtosis']
    
    trade_columns_order = ['No Trades Daily', 'No Long Trades', 'No Short Trades', 'Mean Return','Mean Return (long)', 'Mean Return (short)','Standard Error', 't-Statistic', 'Minimum', '25% Quantile',
                                            'Median', '75% Quantile', 'Maximum', 'Share > 0', 'Standard Deviation', 'Skewness','Kurtosis']
    
    holding_period_analysis_df.columns = ['Minimum', '25% Quantile', 'Median', '75% Quantile', '90% Quantile', '95% Quantile', '99% Quantile',
                                          'Maximum', '% over 2 * timestep', '% over 3 * timestep', '% over 4 * timestep', '% over 5 * timestep']
    holding_period_analysis_cc_df.columns = ['Minimum', '25% Quantile', 'Median', '75% Quantile', '90% Quantile', '95% Quantile', '99% Quantile',
                                          'Maximum', '% over 2 * timestep', '% over 3 * timestep', '% over 4 * timestep', '% over 5 * timestep']
    
    kpi_df.columns          = ['No Trades', 'Mean Return', 'Standard Error', 't-Statistic', 'Minimum', '25% Quantile',
                               'Median', '75% Quantile', 'Maximum', 'Share > 0', 'Standard Deviation', 'Skewness',
                               'Kurtosis']
    kpi_tc_df.columns       = ['No Trades', 'Mean Return', 'Standard Error', 't-Statistic', 'Minimum', '25% Quantile',
                               'Median', '75% Quantile', 'Maximum', 'Share > 0', 'Standard Deviation', 'Skewness',
                               'Kurtosis']
    
    
    kpi_trade_df.columns    = trade_columns
    kpi_trade_df            = kpi_trade_df[trade_columns_order]
    kpi_trade_cc_df.columns    = trade_columns
    kpi_trade_cc_df            = kpi_trade_cc_df[trade_columns_order]

    kpi_trade_pos_df.columns    = trade_columns
    kpi_trade_pos_df            = kpi_trade_pos_df[trade_columns_order]
    kpi_trade_pos_cc_df.columns    = trade_columns
    kpi_trade_pos_cc_df            = kpi_trade_pos_cc_df[trade_columns_order]
    
    kpi_trade_neg_df.columns    = trade_columns
    kpi_trade_neg_df            = kpi_trade_neg_df[trade_columns_order]
    kpi_trade_neg_cc_df.columns    = trade_columns
    kpi_trade_neg_cc_df            = kpi_trade_neg_cc_df[trade_columns_order]
    
    kpi_trade_tc_df.columns     = trade_columns
    kpi_trade_tc_df             = kpi_trade_tc_df[trade_columns_order]
    kpi_trade_tc_cc_df.columns     = trade_columns
    kpi_trade_tc_cc_df             = kpi_trade_tc_cc_df[trade_columns_order]
    
    kpi_trade_pos_tc_df.columns    = trade_columns
    kpi_trade_pos_tc_df            = kpi_trade_pos_tc_df[trade_columns_order]
    kpi_trade_pos_tc_cc_df.columns    = trade_columns
    kpi_trade_pos_tc_cc_df            = kpi_trade_pos_tc_cc_df[trade_columns_order]
    
    kpi_trade_neg_tc_df.columns    = trade_columns
    kpi_trade_neg_tc_df            = kpi_trade_neg_tc_df[trade_columns_order] 
    kpi_trade_neg_tc_cc_df.columns    = trade_columns
    kpi_trade_neg_tc_cc_df            = kpi_trade_neg_tc_cc_df[trade_columns_order]

    
    holding_period_analysis_df.index = ['k = ' + str(k) + ' Step ' + str(s) for s in steps for k in top]
    holding_period_analysis_cc_df.index = ['k = ' + str(k) + ' Step ' + str(s) for s in steps for k in top]
    kpi_df.index                = ['k = ' + str(k) + ' Step ' + str(s) for s in steps for k in top]
    kpi_trade_df.index          = ['k = ' + str(k) + ' Step ' + str(s) for s in steps for k in top]
    kpi_trade_pos_df.index      = ['k = ' + str(k) + ' Step ' + str(s) for s in steps for k in top]
    kpi_trade_neg_df.index      = ['k = ' + str(k) + ' Step ' + str(s) for s in steps for k in top]
    kpi_trade_pos_tc_df.index   = ['k = ' + str(k) + ' Step ' + str(s) for s in steps for k in top]
    kpi_trade_neg_tc_df.index   = ['k = ' + str(k) + ' Step ' + str(s) for s in steps for k in top]
    kpi_trade_cc_df.index          = ['k = ' + str(k) + ' Step ' + str(s) for s in steps for k in top]
    kpi_trade_pos_cc_df.index      = ['k = ' + str(k) + ' Step ' + str(s) for s in steps for k in top]
    kpi_trade_neg_cc_df.index      = ['k = ' + str(k) + ' Step ' + str(s) for s in steps for k in top]
    kpi_trade_pos_tc_cc_df.index   = ['k = ' + str(k) + ' Step ' + str(s) for s in steps for k in top]
    kpi_trade_neg_tc_cc_df.index   = ['k = ' + str(k) + ' Step ' + str(s) for s in steps for k in top]
    kpi_tc_df.index             = ['k = ' + str(k) + ' Step ' + str(s) for s in steps for k in top]
    kpi_trade_tc_df.index       = ['k = ' + str(k) + ' Step ' + str(s) for s in steps for k in top]
    kpi_trade_tc_cc_df.index       = ['k = ' + str(k) + ' Step ' + str(s) for s in steps for k in top]
    
    daily_kpi = [kpi_df, kpi_tc_df]
    trade_kpi = [kpi_trade_df, kpi_trade_tc_df]
    pos_trade_kpi = [kpi_trade_pos_df, kpi_trade_pos_tc_df]
    neg_trade_kpi = [kpi_trade_neg_df, kpi_trade_neg_tc_df]
    
    trade_cc_kpi = [kpi_trade_cc_df, kpi_trade_tc_cc_df]
    pos_trade_cc_kpi = [kpi_trade_pos_cc_df, kpi_trade_pos_tc_cc_df]
    neg_trade_cc_kpi = [kpi_trade_neg_cc_df, kpi_trade_neg_tc_cc_df]
    
    kpi_dict = {'Daily KPI': daily_kpi,
                'Trade KPI': trade_kpi,
                'Pos Trade KPI': pos_trade_kpi,
                'Neg Trade KPI': neg_trade_kpi,
                'Holding Period Analysis': holding_period_analysis_df,
                'Daily Total': daily_rets_list,
                'Daily Total TC': daily_rets_tc_list}
    
    kpi_trade_cc_dict = {'Trade KPI': trade_cc_kpi,
                'Pos Trade KPI': pos_trade_cc_kpi,
                'Neg Trade KPI': neg_trade_cc_kpi,
                'Holding Period Analysis': holding_period_analysis_cc_df,
                }
        
    return kpi_dict, kpi_trade_cc_dict, df_list



def calc_daily_ret_new(df, tc=0):
    helper_df = df.copy()
    
    # substract transaction costs and aggregate to daily level    
    returns_list_tc     = np.array([helper_df.iloc[i,3]-tc if helper_df.iloc[i,14] != -1 else 0 for i in range(len(helper_df))])
    helper_df['Return'] = returns_list_tc
    helper_df.index = [pd.to_datetime(helper_df['Exit helper']), helper_df.Coin]
    avg_ts = helper_df['Return'].groupby(level='Exit helper').mean()
    agg_daily = pd.DataFrame(avg_ts.resample('D').apply(lambda x : ((1+x).cumprod()-1).iloc[-1]))
    agg_daily = agg_daily.loc[agg_daily.index > datetime(2018,6,17)]
    agg_daily = agg_daily.loc[agg_daily.index < datetime(2018,9,8)]
    
#    #helper_df['Date']   = [datetime.strptime(date, '%d.%m.%Y') for date in list(helper_df['Date'])]
#    
##    # only take full days and the same days for all
##    helper_df = helper_df[(helper_df['Date'] > datetime.strptime('19.06.2018', '%d.%m.%Y')) & (helper_df['Date'] < datetime.strptime('08.09.2018', '%d.%m.%Y'))]
##    days = [datetime.strptime('20.06.2018', '%d.%m.%Y') + timedelta(days=i) for i in range(80)]
#    # get the relevant df
#    helper_two = helper_df[['Return','Time entry']]
#    
#    # get the average return per timestamp
#    helper_list = []    
#    points_in_time = sorted(list(set(helper_two['Time entry'])))
#    avg_trade_df = pd.DataFrame()
#    
#    for timestamp in points_in_time:        
#        helper_variable = helper_two[helper_two['Time entry'] == timestamp]
#        if len(helper_variable != 0):
#            helper_list.append(np.nanmean(helper_variable['Return']))
#            avg_trade_df = avg_trade_df.append(pd.DataFrame([np.nanmean(helper_variable['Return']), timestamp]).T)
#        else:
#            helper_list.append(0)
#            avg_trade_df = avg_trade_df.append(pd.DataFrame([0, timestamp]).T)
#                           
#    # get the returns as of end of day
#    avg_trade_df.columns = ['Return', 'Date']
#    avg_trade_df.index = pd.to_datetime(avg_trade_df.Date)
#    del(avg_trade_df['Date'])
#    
#    
#    day_end = avg_trade_df.resample('D').apply(lambda x : ((1+x).cumprod()-1).iloc[-1])
    # accumulate the returns
    day_end_returns_cumsum = agg_daily.cumsum()

    day_end_df = pd.DataFrame()
    day_end_df['Date'] = agg_daily.index
    day_end_df.index = pd.to_datetime(day_end_df.Date)
    del(day_end_df['Date'])
    day_end_df['Return'] = agg_daily.Return.values
    
    acc_df = pd.DataFrame()
    acc_df['Date'] = agg_daily.index
    acc_df.index = pd.to_datetime(acc_df.Date)
    del(acc_df['Date'])
    acc_df['Return'] = day_end_returns_cumsum.Return.values + 1
    
    return agg_daily, day_end_returns_cumsum, len(agg_daily), acc_df, day_end_df


def get_long_short_returns(df, tc=0.002):
    long_trades = df.loc[df['Type of trade'] == 'long']
    short_trades = df.loc[df['Type of trade'] == 'short']
    
    cumsum_long = pd.DataFrame(calc_daily_ret_new(long_trades, 0)[0])
    cumsum_long.columns = ['return long']
    cumsum_long_tc = pd.DataFrame(calc_daily_ret_new(long_trades, tc)[0])
    cumsum_long_tc.columns = ['return long']
    cumsum_short = pd.DataFrame(calc_daily_ret_new(short_trades, 0)[0])
    cumsum_short.columns = ['return short']
    cumsum_short_tc = pd.DataFrame(calc_daily_ret_new(short_trades, tc)[0])
    cumsum_short_tc.columns = ['return short']
    
    total_ret = pd.concat([cumsum_long, cumsum_short], axis=1)
    total_ret['return total'] = total_ret.mean(axis=1)
    total_ret_tc = pd.concat([cumsum_long_tc, cumsum_short_tc], axis=1)
    total_ret_tc['return total'] = total_ret_tc.mean(axis=1)
    
    dates = pd.DataFrame(calc_daily_ret_new(long_trades, 0)[3]['Date'])
    total_ret.index = pd.to_datetime(dates.Date)
    total_ret_tc.index = pd.to_datetime(dates.Date)
    
    return total_ret, total_ret_tc

def plot_rets_long_short(list_of_dfs, tc=0.002):
    for i, entry in enumerate(list_of_dfs):
        total_ret, total_ret_tc = get_long_short_returns(entry[2][0], tc)
        ax = plt.subplot(int(len(list_of_dfs) / 2), 2, i+1)
        ax.plot(total_ret_tc.cumsum()+1)

def get_long_short_mean(list_of_dfs, tc=0.002):
    total_ret = 0
    total_ret_tc = 0
    
    for entry in list_of_dfs:
        helper, helper_tc = get_long_short_returns(entry[2][0], tc)
        total_ret += helper.values / 12
        total_ret_tc += helper_tc.values / 12
    
    return total_ret, total_ret_tc

def calc_daily_returns_cross_eva(cross_eva_df, tc=0):
    """ This function calculates the daily returns of the crosssection trading strategy
    Args:
        cross_eva_df:   A dataframe with the results of the crosssection trading, provived by evaluate_performance_crosssection()
        top:            The number of top/flop assets that were traded
        time_step:      The holding period of the assets
    Returns:
        daily_returns:  A list with arrays containing the geometric returns per day
    """
    
    cross_eva_df_copy = cross_eva_df.copy()
    # get the different days
    days = sorted(list(set(cross_eva_df['Date'])), key=sorting)
    # subtracted the transaction costs of those trades that are not "0-filled"
    returns_list_tc     = np.array([cross_eva_df_copy.iloc[i,3]-tc if cross_eva_df_copy.iloc[i,7] != -1 else 0 for i in range(len(cross_eva_df_copy))])
    # update the df
    cross_eva_df_copy['Return'] = returns_list_tc
    # make a list of arrays, each representing the returns made in one time step
    timeframe_daily     = [np.array(cross_eva_df_copy[cross_eva_df['Date'] == day]['Return']) for day in days]
    
    timeframe_daily_returns = []
    for i, tf_daily in enumerate(timeframe_daily):
        if len(tf_daily) < 110:
            helper_len = 1
        else:
            helper_len = 12 
        helper_list = []
        for j in range(helper_len):
            if j < 11:
                helper_list.append(np.mean(tf_daily[j*10:(j+1)*10]))
            else:
                helper_list.append(np.mean(tf_daily[j*10:len(tf_daily)-1]))
        timeframe_daily_returns.append(helper_list)
    
    
    cumprod_daily = [np.cumprod(np.array(timeframe_daily_returns[i])+1) for i in range(len(timeframe_daily_returns))]    
    # get the number of days
    no_days = int(len(cumprod_daily))     
    # cumprod of the returns of each day, to get the reinvested profits of one day
    daily_returns = [cumprod_daily[i][len(cumprod_daily[i])-1] for i in range(len(cumprod_daily))]
    
    total_daily = np.array([cumprod_daily[i][len(cumprod_daily[i])-1] for i in range(len(cumprod_daily))])-1
    total_daily = np.cumsum(total_daily)
    total_daily_helper = total_daily 
    total_df_daily    = pd.DataFrame()
    total_df_daily['Date'] = days
    total_df_daily['Returns'] = np.array(daily_returns)-1
    
    total_df_accum = pd.DataFrame()
    total_df_accum['Date'] = days
    total_df_accum['Accumulated Returns'] = np.array(total_daily)+1
    
    return np.array(daily_returns)-1, no_days, cumprod_daily, total_daily_helper, total_df_daily, total_df_accum    


def analyze_return_per_coin(df, timestep=120):       
    coins = sorted(list(set(df['Coin'])))
    
    per_coin_kpi = np.zeros((len(coins), 18))
    
    for i, c in enumerate(coins):
        helper_df = df[df['Coin'] == c]
        if len(helper_df[helper_df['Top/Flop'] != -1]) == 0:
            per_coin_kpi[i,0] = len(helper_df)
            continue
        else:
            no_trades = len(helper_df)
            helper_df = helper_df[helper_df['Top/Flop'] != -1]
            helper_long = helper_df[helper_df['Type of trade'] == 'long']
            helper_short = helper_df[helper_df['Type of trade'] == 'short']
            per_coin_kpi[i,6:] = get_kpi_array(np.array(helper_df['Return']))
            per_coin_kpi[i,5]  = np.mean(np.array(helper_short['Return']))
            per_coin_kpi[i,4]  = np.mean(np.array(helper_long['Return']))
            per_coin_kpi[i,3]  = len(helper_short)
            per_coin_kpi[i,2]  = len(helper_long)
            per_coin_kpi[i,1]  = len(helper_df[helper_df['Top/Flop'] != -1])
            per_coin_kpi[i,0]  = no_trades
            
    
    per_coin_df = pd.DataFrame(per_coin_kpi)
    per_coin_df.columns = ['No Trades', 'No Executed Trades', 'No. long trades', 'No. short trades', 'Mean Return Long', 'Mean Return Short', 'Mean Return', 'Standard Error', 't-Statistic (NW)', 'Minimum', '25% Quantile',
                           'Median', '75% Quantile', 'Maximum', 'Share > 0', 'Standard Deviation', 'Skewness',
                           'Kurtosis']
    per_coin_df.index = coins
    return per_coin_df.round(decimals=4)

    
def analyze_holding_period(df, top=True, timestep=120):
    
    if top:
        holding_period = np.array(df[df['Top/Flop']!=-1]['Holding time'])
    else: 
        holding_period = np.array(df['Holding time'])

    
    five_times = round(len(holding_period[holding_period > 5 * timestep]) / len(holding_period), 4) * 100
    four_times = round(len(holding_period[holding_period > 4 * timestep]) / len(holding_period), 4) * 100
    three_times = round(len(holding_period[holding_period > 3 * timestep]) / len(holding_period), 4) * 100
    two_times = round(len(holding_period[holding_period > 2 * timestep]) / len(holding_period), 4) * 100

    
    holding_period_data = [np.min(holding_period), np.percentile(holding_period, 25), np.median(holding_period), np.percentile(holding_period, 75), 
                                np.percentile(holding_period, 90), np.percentile(holding_period, 95), np.percentile(holding_period, 99), 
                                np.max(holding_period), two_times, three_times, four_times, five_times]
    
    return holding_period_data
    
    
def sorting(L):
    splitup = L.split('.')
    return splitup[1], splitup[0] 
        
def get_kpi_dict(data_dict, coin):
    kpi_list = []
    
    keys = list(data_dict.keys())
    key  = [k for k in keys if str(k)[0:k.find("_")] == coin][0]
    data = data_dict[key]
    kpi_list.append(data[3]['No. long'] + data[3]['No. short'])
    kpi_list.append(np.mean(data[3]['Returns per trade']))
    kpi_list.append(np.mean(data[3]['Returns long trades']))
    kpi_list.append(np.mean(data[3]['Returns short trades']))
    kpi_list.append(np.std(data[3]['Returns per trade'])/np.sqrt(data[3]['No. long'] + data[3]['No. short']))
    kpi_list.append(ss.ttest_ind(data[3]['Returns per trade'], np.zeros((len(data[3]['Returns per trade']),)))[0])
    kpi_list.append(np.min(data[3]['Returns per trade']))
    kpi_list.append(np.percentile(data[3]['Returns per trade'], q=25))
    kpi_list.append(np.median(data[3]['Returns per trade']))
    kpi_list.append(np.percentile(data[3]['Returns per trade'], 75))
    kpi_list.append(np.max(data[3]['Returns per trade']))
    kpi_list.append(len([x for x in data[3]['Returns per trade'] if x>0]) / len(data[3]['Returns per trade']))
    kpi_list.append(np.std(data[3]['Returns per trade']))
    kpi_list.append(ss.skew(data[3]['Returns per trade']))
    kpi_list.append(ss.kurtosis(data[3]['Returns per trade']))
    return kpi_list
   
    
def do_ttest_all(data_dict):
    keys            = list(data_dict.keys())
    keys_no_time    = [key for key in keys if key[len(key)-7:]=='no_time']
    keys_time       = [key for key in keys if key not in keys_no_time]

    t_test_time     = []
    t_test_notime   = []

    returns_helper_time     = [[],[],[],[],[]]
    returns_helper_no_time  = [[],[],[],[],[]]
    for i in range(len(keys_time)):
        helper_time     = data_dict[keys_time[i]]
        helper_no_time  = data_dict[keys_no_time[i]]
        for j in range(5):
            returns_helper_time[j]      = returns_helper_time[j] + helper_time[j]['Returns per trade']
            returns_helper_no_time[j]   = returns_helper_no_time[j] + helper_no_time[j]['Returns per trade']

    for i in range(5):
        t_test_time.append(ss.ttest_ind(returns_helper_time[i], np.zeros((len(returns_helper_time[i]),)))[0])
        t_test_notime.append(ss.ttest_ind(returns_helper_no_time[i], np.zeros((len(returns_helper_no_time[i]),)))[0])
    
    return t_test_notime, t_test_time
                

def find_sep_char(list_2):
    """ This function finds the separating character in the file names of the given list
    Args:
        list_2: A list with names of files
    Returns:
        dividing_char:  The character that separates the coin from the rest of the file name 
                        (BTC_Bitfinex.csv ---> "_")
    """
    helper_2            = list_2[0]
    underscore_index    = helper_2.find("_")
        
    if underscore_index == -1:
        dividing_char = '.'
    else:
        dividing_char = '_'
    
    return dividing_char
    

def truncate_ts(data):
    """ This function cuts off all zero entries and everything before that of a given array
    Args:
        data: An array, a list, or a series containing the ts to be truncated
    Returns:
        The truncated data
    """
    zero_index = np.where(data==0)
    
    if len(zero_index[0]) == 0:
        return data
    else:
        return np.array(data[zero_index[0][len(zero_index[0])-1]+1:])


def truncate_df_ts(df):
    ts = np.array(df['Time'])
    ts_diff = ts[1:] - ts[:-1]
    ind = np.where(ts_diff > 10000)
    
    if len(ind[0]) != 0:
        return df.iloc[int(ind[len(ind[0])-1])+1:,:]
    else:
        return df    

def mase(training_series, testing_series, prediction_series):
    """
    Computes the MEAN-ABSOLUTE SCALED ERROR forcast error for univariate time series prediction.
    
    parameters:
        training_series: the series used to train the model, 1d numpy array
        testing_series: the test series to predict, 1d numpy array or float
        prediction_series: the prediction of testing_series, 1d numpy array (same size as testing_series) or float
        absolute: "squares" to use sum of squares and root the result, "absolute" to use absolute values.    
    """
    n = training_series.shape[0]
    d = np.abs(  np.diff( training_series) ).sum()/(n-1)

    errors = np.abs(testing_series - prediction_series )
    return errors.mean()/d



def dict_to_df(dictionary):
    """ This function turns a dictionary with only one key into a dataframe
    Args:
        dictionary: The dictionary to be transformed
    Returns:
        df: The resulting dataframe
    """
    keys = list(dictionary.keys())
    values = [dictionary[key] for key in keys]
    df = pd.DataFrame(values)
    df = df.transpose()
    df.columns = keys
    
    return df
    

def pred_to_files(path_result, pred, pred_len, coins, steps, classification=False):
    """ This is a helper function for the one-model-for-all RandomForest approach. It creates files
        with the predictions according to each coin so that the evaluation function can handle them.
    Args:
        path_result: The path of the directory of where to store the resulting files
        pred:           A dict with the predictions, each key being one timestep
        coins:          A list of the coins for which the predictions have been made
        steps:          A list of the timesteps
    Returns:
        The newly created .csv files
    """        
        
    index_helper = [0] + list(np.cumsum(pred_len))


    for i in range(len(coins)):
      
        helper = pred['Forecast horizon ' + str(steps)][int(index_helper[i]):int(index_helper[i+1])]
        
        helper_df = pd.DataFrame(helper)
        helper_df.columns = ['Forecast horizon ' + str(steps)]
        
        if classification:
            helper_df.to_csv(path_result + str(coins[i]) + '_classification.csv')
        else:
            helper_df.to_csv(path_result + str(coins[i]) + '_regression.csv')


def get_train_test_data_mat(path_data, step, test_size=0.3):
    data_files  = [f for f in listdir(path_data) if isfile(join(path_data, f))]
    helper_data = pd.read_csv(path_data + 'BTC_Bitfinex.csv')
    
    len_train = int(len(helper_data)*(1-test_size))
    len_test  = int(len(helper_data)*test_size)
    
    close_mat_train = np.zeros((len(data_files), len_train))
    vol_mat_train   = np.zeros((len(data_files), len_train))
    ret_mat_train   = np.zeros((len(data_files), len_train-step))
    
    close_mat_test = np.zeros((len(data_files), len_test))
    vol_mat_test   = np.zeros((len(data_files), len_test))
    ret_mat_test   = np.zeros((len(data_files), len_test-step))
    
    for i in range(len(data_files)):
        helper = pd.read_csv(path_data + data_files[i])
        close = truncate_ts(np.array(helper['Close']))
        
        training_close, test_close  = train_test_split(close, test_size=test_size, shuffle=False)
        training_vol, test_vol      = train_test_split(np.array(helper['VolumeFrom']), test_size=test_size, shuffle=False)
        
        close_mat_train[i,-len(training_close):]    = training_close
        close_mat_test[i,-len(test_close):]         = test_close
        vol_mat_train[i,-len(training_vol):]        = training_vol
        vol_mat_test[i,-len(test_vol):]             = test_vol
        ret_mat_train[i,-len(training_close)-step:] = get_returns(training_close, step, False)
        ret_mat_test[i,-len(test_close)-step:]      = get_returns(test_close, step, False)
        
    return close_mat_train, close_mat_test, vol_mat_train, vol_mat_test, ret_mat_train, ret_mat_test
    

def convert_target_df(df):
    helper = pd.DataFrame(df['target'])
    minutely_median = pd.DataFrame(helper.groupby(level='Date').median())
    minutely_median.columns = ['minutely median']
    
    helper = helper.join(minutely_median)
    helper['binary target'] = 0
    helper['binary target'].loc[helper['target']>=helper['minutely median']] = 1
    
    df.target = helper['binary target']
    
    return df
    
def convert_target_classification(train_mat, len_list):
    """ This function converts the regression target of a given feature matrix into a binary, 
        that is 0/1 target. The decision which target to put into which class is based on the cross-
        sectional median of the target. Meaning also, that this function assumes to get a matrix with multiple
        feature inputs. It is therefore used in the one-model-for-all coins approach.
    Args:
        train_mat: The matrix with the features (the target must be in the right-most column)
        len_list:  A  list of the different lenghts of feature blocks so that the function can build a cross-
                    sectional median of those
    Returns:
        train_mat: The given matrix but with a binary target
    """
    train_mat   = np.nan_to_num(train_mat)
    n_cols_mat  = len(train_mat[0,:])
    max_len     = max(len_list)
    
    returns_mat = np.empty((len(len_list), max_len))
    returns_mat[:] = np.nan
    
    for i in range(len(len_list)):
        returns_mat[i,-len_list[i]:] = train_mat[sum(len_list[:i]):sum(len_list[:i])+len_list[i], n_cols_mat-1]


    for i in range(max_len):
        helper = [returns_mat[j,i] for j in range(len(len_list)) if np.isnan(returns_mat[j,i]) == False]
        helper_med = np.median(helper)
        
        for j in range(len(len_list)):
            if returns_mat[j,i] >= helper_med and np.isnan(returns_mat[j,i]) == False:
                returns_mat[j,i] = 1
            elif returns_mat[j,i] < helper_med and np.isnan(returns_mat[j,i]) == False:
                returns_mat[j,i] = 0
    
    helper_list = []
    for i in range(len(len_list)):
        helper_list.append(returns_mat[i,-len_list[i]:])
    
    
    helper_list = list(itertools.chain(*helper_list))
    train_mat[:, n_cols_mat-1] = helper_list
    
    return train_mat
        
    
def sparse_argsort(arr):
    """ This function performs an argsort of an array while ignoring zero values in it
    Args: 
        arr: The array to be sorted
    Returns:
        indices: The indices of the values in ascending order, ignoring zero values
    """
    indices = np.nonzero(arr)[0]
    return indices[np.argsort(arr[indices])]    


def match_lists_helper(list_one, list_two):
    """ This function finds the coins which are in both given lists
    Args:
        list_one: The first list of files
        list_two: The second list of files
    Returns:
        coins: A list with the coins which appear in both input lists
    """
    
    sep_char_one  = find_sep_char(list_one)
    sep_char_two = find_sep_char(list_two)
    
    files_one_helper   = [str(f)[0:str(f).find(sep_char_one)] for f in list_one]
    files_two_helper    = [str(f)[0:str(f).find(sep_char_two)] for f in list_two]
    
    list_one    = [f for f in list_one if str(f)[0:str(f).find(sep_char_one)] in files_two_helper]       
    list_two    = [f for f in list_two if str(f)[0:str(f).find(sep_char_two)] in files_one_helper]
    
    coins = [str(f)[0:str(f).find(sep_char_one)] for f in list_one]
    
    return coins


def list_of_sets_helper(list_of_sets):
    """ This function returns the intersection of all sets given in a list
    Args:
        list_of_sets: The list of sets of which the intersection needs to be found
    Returns:
        list: A list with the intersection of the entries of the input
    """
    while len(list_of_sets) > 1:
        helper = list_of_sets[0].intersection(list_of_sets[1])
        list_of_sets = list_of_sets[1:]
        list_of_sets[0] = helper
    
    return list(list_of_sets[0])


def match_files(path_one, path_two):
    """ This function matches the files given in the input paths
    Args:
        path_one: The path with the data files (usually)
        path_two: The path with the prediction files
    Returns:
        The coins in the intersection of both input paths
    """
        
    # getting the first file
    first_files  = [f for f in listdir(path_one) if isfile(join(path_one, f))]
    # get the separation character
    sep_char_files = find_sep_char(first_files)
    
    # checking how many predicition files there are and reading them in
    if type(path_two) == list and len(path_two) > 1:
        second_files   = [[f for f in listdir(path_two[i]) if isfile(join(path_two[i], f))] for i in range(len(path_two))]
    elif type(path_two) == list and len(path_two) == 1:
        second_files   = [f for f in listdir(path_two[0]) if isfile(join(path_two[0], f))]
    elif type(path_two) == str:
        second_files   = [f for f in listdir(path_two) if isfile(join(path_two, f))]
    
    # check if there are several lists in the second files list
    if any(isinstance(el, list) for el in second_files):
        coins_helper = [[] for l in range(len(second_files))]
        # get the separation chars for the pred files
        sep_char_preds = [find_sep_char(second_files[i]) for i in range(len(second_files))]
        # find the intersection of all the pred files with the data files
        for i in range(len(second_files)):
            coins_helper[i] = set(match_lists_helper(first_files, second_files[i]))    
        coins       = list_of_sets_helper(coins_helper)   
        # get the relevant prediction files            
        pred_files  = [[f for f in second_files[i] if str(f)[0:str(f).find(sep_char_preds[i])] in coins] for i in range(len(second_files))]
    else:
        coins = match_lists_helper(first_files, second_files)
        sep_char_preds = find_sep_char(second_files)
        pred_files = [f for f in second_files if str(f)[0:str(f).find(sep_char_preds)] in coins]
    # get the relevant data files   
    data_files  = [f for f in first_files if str(f)[0:str(f).find(sep_char_files)] in coins]
    
    return data_files, pred_files, coins
    
def scaler_func(arr):
    """ This function scales an input array to a 0 - 100% scale
    Args: 
        arr: The array to be scaled
    Returns:
        scaled_arr: The scaled array
    """
    helper = [arr[i] for i in range(len(arr)) if np.isnan(arr[i]) == False]
    helper_min = min(helper)
    helper_max = max(helper)
    
    scaled_arr = np.array([(arr[i] - helper_min) / (helper_max - helper_min) for i in range(len(arr))])
    
    return scaled_arr
    
def get_quantiles(arr, quants=[90,95,99]):
    """ This function returns the specified quantiles of a given array
    Args:
        arr:    The array of which the quantiles are wanted
        quants: The quantiles of interest. Default is [90,95,99]
    Returns:
        arr_quants: A list with the specified quantiles of the given array
    """
    arr_quants = []
    for quantile in quants:
        quant = np.percentile(arr, q=quantile)
        arr_quants.append(quant)
    
    return arr_quants

def get_quants_from_df(df, quants=[90,95,99]):
    """ This function returns the quantiles of holding period of the trades in a dataframe 
        as created by code.evaluate_performance_crosssection()
    Args:
        df: The dataframe with the data
        quants: The quantiles of interest. Default is [90,95,99]
    Returns:
        quantiles: A list with the specified quantiles
    """
    time_now    = np.array(df['Time exit'])
    time_bought = np.array(df['Time entry'])
    
    quantiles = get_quantiles(time_now-time_bought, quants=quants)
    
    return quantiles

def create_quants_df(df_list, quants=[90,95,99]):
    """ This function creates a dataframe with the quantiles of the holding periods of several dataframes given
    Args:
        df_list:    A list with dataframes as created by code.evaluate_performance_crosssection()
        quants:     The quantiles of interest. Default is [90,95,99]
    Returns:
        quant_df: A dataframe with the desired quantiles
    """
    quant_mat = np.zeros((len(df_list),len(quants)))
    
    i = 0
    for df in df_list:
        quant_mat[i,:] = get_quants_from_df(df, quants=quants)
        i += 1
    
    quant_df = pd.DataFrame(quant_mat)
    quant_df.columns = [str(q) + '% quantile' for q in quants]
    
    return quant_df

def count_index_exchange(df_list, coins_list):
    
    helper_mat = np.zeros((int(len(df_list)/5),5))
    
    i = 0
    j = 0
    l = 0
    for df in df_list:
        helper = df['Coin']
        helper_ex = [coin for coin in helper if coin in coins_list]
        helper_mat[l,j] = round(len(helper_ex)/len(helper),2)
        i += 1
        j = i % 5
        if j == 0:
            l += 1
    
    df = pd.DataFrame(helper_mat)
    df.columns = ['k = 10', 'k = 5', 'k = 3', 'k = 2', 'k = 1']
    
    return df
    

def staleness_filter(arr, step, option='returns', intensity=0.5):
    if option == 'returns':
        returns = get_returns(arr, step=step, trading_period=False)
        
        returns_masked = [1 if returns[i] != 0 else 0 for i in range(len(returns))]
        
        if sum(returns_masked) > len(returns_masked) * intensity:
            return True
        else:
            return False
    elif option == 'volume':
        vols_masked = [1 if arr[i] > 1.0 else 0 for i in range(len(arr))]
        
        if sum(vols_masked) > len(arr) * intensity:
            return [True, sum(vols_masked) / len(arr)]
        else:
            return [False, sum(vols_masked) / len(arr)]
    else:
        return True



def plt_ex(data, coin):    
    
    plt.close()
    helper = data.loc[coin]
    ex = list(set(helper.index.get_level_values(0)))
    
    df_list = []
    
    for e in ex:
        df_list.append(helper.loc[str(e)])
    
    for i in range(len(df_list)):
        ax = plt.subplot(len(df_list), 1, i+1)
        ax.plot([j for j in range(len(df_list[i]))], df_list[i]['Close'])
        start_date = datetime.fromtimestamp(np.min(df_list[i]['Time'])).strftime('%d%B')
        end_date = datetime.fromtimestamp(np.max(df_list[i]['Time'])).strftime('%d%B')
        vol = '{:,}'.format(round(np.sum(df_list[i]['VolumeFrom']),2))
        ax.set_title(coin + ' on ' + str(ex[i]) + ' from ' + start_date + ' until ' + end_date + ' with volume ' + str(vol))
    plt.show()


def calc_avg_vola(path_data, steps=[1,10,30,60,120]):
    data_files  = [f for f in listdir(path_data) if isfile(join(path_data, f))]
    data = [pd.read_csv(path_data + d) for d in data_files]
  
    vola_list = [] 
    for s in steps:      
        helper_list = []
        for d in data:
            helper = get_returns(np.array(d['Close']), s, False)
            helper_list.append(np.std(helper))
        vola_list.append(round(np.mean(helper_list),4))
    
    vola_df = pd.DataFrame(vola_list)
    vola_df = vola_df.transpose()
    vola_df.columns = ['Step ' + str(s) for s in steps]
    
    return vola_df
    
    
    
def calc_corr(path_data, corr_of='returns', time_step=120):
    
    files   = [f for f in listdir(path_data) if isfile(join(path_data, f))]
    data    = [pd.read_csv(path_data + f) for f in files]
    coins   = [str(f)[:str(f).find("_")] for f in files]
    
    prices  = [np.array(df['Close']) for df in data]     
    returns = [get_returns(prices[i], time_step, False) for i in range(len(prices))]
    
    min_len = np.min([len(p) for p in returns])
    
    returns_mat = np.zeros((len(returns), min_len))
    
    for i in range(len(returns)):
        returns_mat[i,:] = returns[i][-min_len:]
    
    corr = np.corrcoef(returns_mat)
    corr_df = pd.DataFrame(corr)
    corr_df.columns = coins
    corr_df.index   = coins
    
    return corr_df

def get_google_trends_mat(path_gt):
    gt_files    = [f for f in listdir(path_gt) if isfile(join(path_gt, f))]
    coins       = [str(f)[:str(f).find("_")] for f in gt_files]
    data_gt     = [pd.read_csv(path_gt + file) for file in gt_files]
    
    
    gt_mat = np.zeros((len(coins),len(data_gt[0])))
    
    for i, d in enumerate(data_gt):
        gt_mat[i,1:] = get_returns(np.array(d[coins[i] + ' price']), 1, False)
    
    return gt_mat

def get_hourly_data_mat(path, factor=60):
    files    = [f for f in listdir(path) if isfile(join(path, f))]
    coins    = [str(f)[:str(f).find("_")] for f in files]
    data     = [pd.read_csv(path + file) for file in files]
    data_hourly = [d[::factor] for d in data]
    start    = [1 if d['Datetime'][0] == '2018-01-05 00:01:00' else 0 for d in data_hourly]

    data_mat = np.zeros((len(coins), max([len(d) for d in data_hourly])))
    data_mat[:] = np.nan
    
    for i, d in enumerate(data_hourly):
        if start[i] == 1:
            data_mat[i,:len(d)-1] = get_returns(np.array(d['Close']), 1, False)
        else:
            data_mat[i,-len(d)+1:] = get_returns(np.array(d['Close']), 1, False)
    
    return data_mat

def build_crypto_index(path):
    files    = [f for f in listdir(path) if isfile(join(path, f))]
    coins    = [str(f)[:str(f).find("_")] for f in files]
    data_list   = [pd.read_csv(path + file) for file in files]
    data_list   = [d.sort_values(by='Time') for d in data_list]
    data_list   = [truncate_df_ts(df) for df in data_list]
    
    
    prices_mat = np.zeros((len(coins), 354240))
    prices_mat[:] = np.nan
    returns_mat = np.zeros((len(coins), 354239))
    returns_mat[:] = np.nan
    cumprod_mat = np.zeros((len(coins), 354239))
    cumprod_mat[:] = np.nan
    
    for i, df in enumerate(data_list):
        prices_mat[i,-len(df):]     = np.array(df['Close'])
        returns_helper = np.array(df['Close'])[1:] / np.array(df['Close'])[:-1] -1
        returns_mat[i, -len(df)+1:] = returns_helper 
        cumprod_mat[i, -len(df)+1:] = np.cumprod(returns_helper+1)
        
    
    crypto_index = np.ones((354240))
    crypto_returns = np.ones(354240)
    
    for i in range(np.shape(cumprod_mat)[1]):
        crypto_index[i+1] = np.nanmean(cumprod_mat[:,i])
        crypto_returns[i+1] = np.nanmean(returns_mat[:,i])
        
    return cumprod_mat, crypto_index, crypto_returns, returns_mat

import matplotlib.pyplot as plt

def calc_buy_and_hold_kpi(path_file, train_test_split=2/3, forecast_horizon=120):
    file = pd.read_csv(path_file)
    file.index = pd.to_datetime(file.Datetime)
    file.index.names = ['Date']
    
    # extract the closing prices
    close_prices = file.Close
    plt.plot(close_prices)
    print(close_prices.describe())
    
    # extract the trading period
    all_dates = file.index.get_level_values('Date').drop_duplicates().values
    idx_first_date_trading = int(len(all_dates) * train_test_split)
    
    trading_prices = close_prices[close_prices.index.get_level_values('Date')>=all_dates[idx_first_date_trading]]
    plt.plot(trading_prices)
    print(trading_prices.describe())
    
    # add legend
    plt.legend(['All prices', 'Trading period'])
    
    trade_returns = get_returns(trading_prices, forecast_horizon, False)
    # calc the kpi
    kpi_trade = get_kpi_array(trade_returns)   
    # add the number of "trades"
    kpi_trade = [int(len(trade_returns))] + kpi_trade
    # calc the kpi
    kpi_daily = get_kpi_array(trading_prices.resample('D').apply(lambda x: x[-1] / x[0] - 1))
    # add the number of "trading days"
    kpi_daily = [int(len(trading_prices) / 1440)] + kpi_daily
    
    # format the dataframes
    kpi_df = pd.DataFrame()
    kpi_df['Daily'] = kpi_daily
    kpi_df['Trade'] = kpi_trade
    kpi_df.index = ['No trades / days', 'Mean return', 'Standard error', 't-Statistic', 'Minimum', '25% Quantile', 'Median', '75% Quantile', 'Maximum', 'Share > 0',
                    'Standard dev.', 'Skewness', 'Kurtosis']
    
    buy_and_hold_complete       = round(close_prices[-1] / close_prices[0] - 1, 6)
    buy_and_hold_trading_period = round(trading_prices[-1] / trading_prices[0] - 1, 6)
    print("Buy-and-hold during the trading period yields " + str(buy_and_hold_trading_period * 100) + "% return")
    print("Buy-and-hold duirng the complete time period yields " + str(buy_and_hold_complete * 100) + "% return")
    return kpi_df.round(decimals=6), buy_and_hold_trading_period, buy_and_hold_complete, trade_returns, close_prices
    
    
    
     
    

def build_latex_df(list_of_dicts, path_csv, mkt=None, k=3, model_names = ['BR', 'RF_ind', 'LR', 'RF_univ']):
    
    if mkt != None:
        daily_mat = np.zeros((12, len(list_of_dicts)*2 +1))
    else:
        daily_mat = np.zeros((12, len(list_of_dicts)*2))
    trade_mat = np.zeros((14, len(list_of_dicts)*2))

    
    for i, res_dict in enumerate(list_of_dicts):
        daily_mat[:,i] = np.array(res_dict[0]['Daily KPI'][0].values)[5-k,1:]
        daily_mat[:,i+len(list_of_dicts)] = np.array(res_dict[0]['Daily KPI'][1].values)[5-k,1:]
        
        trade_mat[:,i] = np.array(res_dict[1]['Trade KPI'][0].values)[5-k,3:]
        trade_mat[:,i+len(list_of_dicts)] = np.array(res_dict[1]['Trade KPI'][1].values)[5-k,3:]
        
        helper_df = res_dict[0]['Daily Total'][5-k][1]
        helper_df.index = helper_df['Date']
        helper_df = helper_df.drop('Date', axis=1)
        helper_df.to_csv(path_csv + str(i+1) + '_daily_rets_' + model_names[i][:2] + '.csv')
        
        helper_df_tc = res_dict[0]['Daily Total TC'][5-k][1]
        helper_df_tc.index = helper_df_tc['Date']
        helper_df_tc = helper_df_tc.drop('Date', axis=1)
        helper_df_tc.to_csv(path_csv + str(i+5) + '_daily_rets_' + model_names[i][:2] + '_tc.csv')
    
    if mkt != None:
        daily_mat[:, len(list_of_dicts)*2] = get_kpi_array(mkt)
        
        
    daily_df = pd.DataFrame(daily_mat)
    
    if mkt != None:
        daily_df.columns = model_names * 2 + ['MKT']
    else:
        daily_df.columns = model_names * 2
        
    daily_df.index = list(list_of_dicts[0][0]['Daily KPI'][0].columns)[1:]
    daily_df = daily_df.round(decimals = 4)
    
    trade_df = pd.DataFrame(trade_mat)
    trade_df.columns = model_names * 2
    trade_df.index = list(list_of_dicts[0][0]['Trade KPI'][0].columns)[3:]
    trade_df = trade_df.round(decimals = 4)
    
    
    return daily_df, trade_df


def build_risk_annu_df(path_csv):

    helper_risk = pd.read_csv(path_csv + "risk_metrics.csv").values[:,1:]
    helper_risk_df = pd.DataFrame(helper_risk).round(decimals=4)
    helper_risk_df.index = ['Hist. VaR 1%', 'Hist. CVaR 1%', 'Hist. VaR 5%', 'Hist. CVaR 5%', 'Maximum drawdown', 'Calmar ratio']
    
    helper_ann = pd.read_csv(path_csv + "annualized.csv").values[:,1:]
    helper_ann_df = pd.DataFrame(helper_ann).round(decimals=4)
    helper_ann_df.index = ['Return p.a.', 'Excess return p.a.', 'Standard dev. p.a.', 'Downside dev.', 'Sharpe ratio', 'Sortino ratio']
    
    return helper_risk_df, helper_ann_df
        

def get_average_results(list_of_dicts, model_name, k_list = [5,4,3,2,1], filter_int=25, printing=False):
    average_daily_returns_mat = np.zeros((82,len(k_list)))
    average_daily_returns_tc_mat = np.zeros((82,len(k_list)))
    
    all_trades = pd.DataFrame()
    len_helper = [0 for k in k_list]
    dates = list(list_of_dicts[0][0]['Daily Total'][0][0].index)
    if len(dates) == 83:
        dates = dates[1:]
        
        
    for result in list_of_dicts:
        all_trades = all_trades.append(result[2])

        for i in range(len(result[0]['Daily Total'])):
            
            if len(np.array(result[0]['Daily Total'][i][1]['Return'])) == 82:
                average_daily_returns_mat[:,i] += np.array(result[0]['Daily Total'][i][1]['Return'])
                average_daily_returns_tc_mat[:,i] += np.array(result[0]['Daily Total TC'][i][1]['Return'])
            else:
                average_daily_returns_mat[:,i] += np.array(result[0]['Daily Total'][i][1]['Return'])[1:]
                average_daily_returns_tc_mat[:,i] += np.array(result[0]['Daily Total TC'][i][1]['Return'])[1:]
            len_helper[i] += 1 
    

    for i, x in enumerate(len_helper):  
        average_daily_returns_mat[:, i] = average_daily_returns_mat[:, i] / x
        if printing:
            csv_helper = pd.DataFrame(average_daily_returns_mat[:, i])
            csv_helper.index = dates
            csv_helper.to_csv('C:/Users/adein/Desktop/DailyRets/CC/Averages_Tens/robust/fixed_filter/' + str(filter_int) + '/' + str(i) +'_daily_rets_' + model_name + '.csv')
        
        average_daily_returns_tc_mat[:, i] = average_daily_returns_tc_mat[:, i] / x
        if printing:
            csv_helper_tc = pd.DataFrame(average_daily_returns_tc_mat[:, i])
            csv_helper_tc.index = dates
            csv_helper_tc.to_csv('C:/Users/adein/Desktop/DailyRets/CC/Averages_Tens/robust/fixed_filter/' + str(filter_int) + '/' + str(i+4) + '_daily_rets_' + model_name + '_tc.csv')
    
    return average_daily_returns_mat, average_daily_returns_tc_mat, all_trades, dates


def build_average_holding_time_df(list_of_dicts,  k, model_names):
    
    helper_matrix = np.zeros((12, len(list_of_dicts)))
    
    
    for i, list_of_tuples in enumerate(list_of_dicts):
        helper_max = []
        for tuples in list_of_tuples:
            helper_matrix[:,i] += np.array(analyze_holding_period(tuples[2][5-k])) / 12
            helper_max.append(tuples[0]['Holding Period Analysis'].values[5-k,7])
        helper_matrix[7,i] = np.max(helper_max)
    
    
    avg_df = pd.DataFrame(helper_matrix)
    avg_df.index = list_of_dicts[0][0][0]['Holding Period Analysis'].columns
    avg_df.columns = model_names

    return avg_df.round(decimals=4)

def holding_period_analysis_all(list_of_dfs, k, model_names = ['BR', 'RF_ind', 'LR', 'RF_univ']):
    
    helper_matrix = np.zeros((12, len(list_of_dfs)))
    list_dfs = [[df[2] for df in dfs] for dfs in list_of_dfs]
    
    for i, tuples in enumerate(list_dfs):
        helper_df = pd.DataFrame()
        for dfs in tuples:
            helper_df = helper_df.append(dfs[5-k])
        
        helper_matrix[:,i] = np.array(analyze_holding_period(helper_df))
    
    holding_df = pd.DataFrame(helper_matrix)
    holding_df.index = list_of_dfs[0][0][0]['Holding Period Analysis'].columns
    holding_df.columns = model_names

    return holding_df.round(decimals=4)



    
def get_average_trade_results(list_of_dfs, k, model_names = ['BR', 'RF_ind', 'LR', 'RF_univ']):
    average_trades = np.zeros((17, len(list_of_dfs)*2))    
    
    list_dfs = [[df[2] for df in dfs] for dfs in list_of_dfs]
    
    for i, tuples in enumerate(list_dfs):
        helper_df = pd.DataFrame()
        for dfs in tuples:
            helper_df = helper_df.append(dfs[5-k])

        trades_df = helper_df[helper_df['Top/Flop']!=-1]
        tc_df     = trades_df.copy()
        average_trades[0,i] = int(len(trades_df) / len(tuples) * 12)
        average_trades[1,i] = int(len(trades_df[trades_df['Type of trade']=='long']) / len(tuples) * 12)
        average_trades[2,i] = int(len(trades_df[trades_df['Type of trade']=='short']) / len(tuples) * 12)
        average_trades[3,i] = np.mean(trades_df['Return'])
        average_trades[4,i] = np.mean(trades_df[trades_df['Type of trade']=='long']['Return'])
        average_trades[5,i] = np.mean(trades_df[trades_df['Type of trade']=='short']['Return'])
        average_trades[6:,i] = get_kpi_array(np.array(trades_df['Return']))[1:]
        
        tc_df['Return'] = np.array(trades_df['Return']) - 0.002
        average_trades[0,i+len(list_of_dfs)] = int(len(trades_df) / len(tuples) * 12)
        average_trades[1,i+len(list_of_dfs)] = int(len(trades_df[trades_df['Type of trade']=='long']) / len(tuples) * 12)
        average_trades[2,i+len(list_of_dfs)] = int(len(trades_df[trades_df['Type of trade']=='short']) / len(tuples) * 12)
        average_trades[3,i+len(list_of_dfs)] = np.mean(tc_df['Return'])
        average_trades[4,i+len(list_of_dfs)] = np.mean(tc_df[tc_df['Type of trade']=='long']['Return'])
        average_trades[5,i+len(list_of_dfs)] = np.mean(tc_df[tc_df['Type of trade']=='short']['Return'])
        average_trades[6:,i+len(list_of_dfs)] = get_kpi_array(np.array(tc_df['Return']))[1:]
        
        
    
    avg_df = pd.DataFrame(average_trades).round(decimals = 4)
    avg_df.index = ['No. trades total', 'No. long trades', 'No short trades', 'Mean return','Mean return (long)', 'Mean return (short)','Standard error', 't-Statistic', 'Minimum', '25% Quantile',
                                            'Median', '75% Quantile', 'Maximum', 'Share > 0', 'Standard dev.', 'Skewness','Kurtosis']
    
    avg_df.columns = model_names * 2
    
    return avg_df

                    
    
def build_avg_df(lists_of_dicts, k, mkt=None, model_names = ['BR', 'RF_ind', 'LR', 'RF_univ'], filter_int=25, printing=False):
    
    if mkt != None:
        helper_mat = np.zeros((12,len(model_names)*2 +1))
    else:
        helper_mat = np.zeros((12,len(model_names)*2))
        
        
    for i, list_dicts in enumerate(lists_of_dicts):
        helper = get_average_results(list_dicts, model_names[i], k_list=[k], filter_int=filter_int, printing=printing)
        helper_mat[:,i] = get_kpi_array(helper[0][:,5-k])
        helper_mat[:,i+len(model_names)] = get_kpi_array(helper[1][:,5-k])
    
    if mkt != None:
        helper_mat[:,len(model_names)*2] = get_kpi_array(mkt)
        avg_df = pd.DataFrame(helper_mat)
        avg_df.columns = model_names * 2 + ['MKT']
    else:
        avg_df = pd.DataFrame(helper_mat)
        avg_df.columns = model_names * 2
        avg_df.columns = model_names * 2


    avg_df.index = ['Mean return', 'Standard error', 't-Statistic', 'Minimum', '25% Quantile',
                      'Median', '75% Quantile', 'Maximum', 'Share > 0', 'Standard dev.', 'Skewness',
                      'Kurtosis']
    
    return avg_df.round(decimals = 8)
    

def plot_cumsum(list_of_averages, dates):
    
    for i in range(1):
        for j, avgs in enumerate(list_of_averages):
            avgs_df = pd.DataFrame(np.cumsum(avgs[0][:,i])+1)
            #avgs_df.index = dates
            
            avgs_df_tc = pd.DataFrame(np.cumsum(avgs[1][:,i])+1)
            #avgs_df_tc.index = dates
            
            #plt.subplot(5,2,1+i*2)
            plt.plot(avgs_df_tc, linewidth=5)
            plt.xticks(fontsize=40)
            plt.yticks(fontsize=40)
            plt.ylim(0.7,1.3)
            plt.xticks([0,16,32,48,64], ['20/06', '06/07', '22/07', '10/08', '26/08'])
            #plt.subplot(5,2,2+i*2)
            #plt.plot(avgs_df_tc)
        plt.legend(['BR', 'RF_ind', 'LR', 'RF_univ'], fontsize=20, loc=2)
            

def plot_avg_cumsum(list_tuples):
    dates = pd.DataFrame(list_tuples[0][0][0]['Daily Total TC'][0][0]['Date'])
    
    helper_mat = np.zeros((80,4))
    for i, tuple_list in enumerate(list_tuples):
        for tuples in tuple_list:
            helper_mat[:,i] += tuples[0]['Daily Total TC'][0][1]['Return'] / len(tuple_list)
    
    avg_df = pd.DataFrame(helper_mat)
    avg_df.columns = ['BR', 'RF_ind', 'LR', 'RF_univ']
    avg_df.index = pd.to_datetime(dates.Date)
    
    plt.plot(avg_df.cumsum()+1)
    plt.legend(['BR', 'RF_ind', 'LR', 'RF_univ'], fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    

def dist_mean(list_of_dicts, k):
    helper_list = []
    helper_list_tc = []
    for entry in list_of_dicts:
        helper_list.append(entry[0]['Daily KPI'][0].iloc[5-k,1])
        helper_list_tc.append(entry[0]['Daily KPI'][1].iloc[5-k,1])
        
    dist_mat = np.zeros((4,2))
    dist_mat[:,0] = [np.min(helper_list),np.mean(helper_list), np.median(helper_list), np.max(helper_list)]
    dist_mat[:,1] = [np.min(helper_list_tc),np.mean(helper_list_tc), np.median(helper_list_tc), np.max(helper_list_tc)]

    return dist_mat


def compare_dist_all(list_of_dists, k):
    helper_mat = np.zeros((4,8))
    for i, entry in enumerate(list_of_dists):
        helper_mat[:,i] = dist_mean(entry, k)[:,0]
        helper_mat[:,i+4] = dist_mean(entry, k)[:,1]
        
    helper_df = pd.DataFrame(helper_mat)
    helper_df.index = ['Minimum', 'Mean', 'Median', 'Maximum']
    helper_df.columns = ['BR', 'RF_ind', 'LR', 'RF_univ'] + ['BR tc', 'RF_ind tc', 'LR tc', 'RF_univ tc']
    
    return helper_df.round(decimals=4)


def check_corrs(path_data):
    files = [f for f in listdir(path_data) if isfile(join(path_data, f))]
    rets_mat = np.zeros((40,354120))
    for i, file in enumerate(files):
        helper = pd.read_csv(path_data + file)
        close = np.array(helper['Close'])
        rets = get_returns(close, 120, False)
        rets_mat[i, -(len(rets)):] = rets
    
    for j in range(3000, 354120, 1440):
        print(j)
        corrs = np.corrcoef(rets_mat[:,j-3000:j])
        print(np.count_nonzero(np.isnan(corrs[:,0])))

