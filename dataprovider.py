import pandas as pd
import numpy as np
import datetime as dt
import os



class CryptoMinuteDataProvider:
    def __init__(self, path_to_raw_data):
        self._raw_data = None
        self._path_to_raw_data = path_to_raw_data


    def _load_data(self):
        path_to_raw_data = self._path_to_raw_data
        files = os.listdir(path_to_raw_data)
        try:
            raw_data = pd.read_hdf(path_to_raw_data + 'raw_data.h5')
        except:
            print('[INFO] Rebuilding raw data from CSVs')
            first_date = '2018-01-05 00:01:00'
            last_date = '2018-09-08 00:00:00'
            datetime_range = pd.date_range(first_date, last_date, freq='min')
            datetime_range = pd.DataFrame(datetime_range)
            datetime_range.columns = ['Date']
            datetime_range.set_index('Date', inplace=True)
            datetime_range['filled'] = 'yes'
            raw_data = []
            for a_file in files:
                try:
                    a_df = pd.read_csv(path_to_raw_data + a_file)
                    a_df.Datetime = pd.to_datetime(a_df.Datetime)
                    coin_name = a_file.replace('.csv', '')
                    a_df.set_index(['Datetime'], inplace=True)
                    a_df.index.names = ['Date']
                    a_df = datetime_range.join(a_df)
                    a_df.Close = a_df.Close.ffill()
                    a_df.loc[a_df.Open.isnull(), 'Open'] = a_df.loc[a_df.Open.isnull()].Close
                    a_df.loc[a_df.High.isnull(), 'High'] = a_df.loc[a_df.High.isnull()].Close
                    a_df.loc[a_df.Low.isnull(), 'Low'] = a_df.loc[a_df.Low.isnull()].Close
                    a_df.VolumeFrom = a_df.VolumeFrom.replace(np.NAN, 0)
                    a_df.VolumeTo = a_df.VolumeTo.replace(np.NAN, 0)
                    a_df['Stock'] = coin_name
                    a_df.set_index('Stock', append=True, inplace=True)
                    del(a_df['filled'])
                    print('[INFO] Loading', a_file)
                    raw_data.append(a_df)
                except:
                    pass
            raw_data = pd.concat(raw_data, axis=0)
            raw_data = raw_data.sort_index(0, level='Date')
            print('[INFO] Raw data successfully read from disk')
            raw_data.columns = ['ad_index', 'open', 'high', 'low', 'close', 
                                'volume_from', 'volume_to', 'time']
            del(raw_data['ad_index'])
            del(raw_data['time'])
            raw_data.to_hdf(path_to_raw_data + 'raw_data.h5', key='data')
        self._raw_data = raw_data
    
    def get_full_raw_data(self):
        if self._raw_data is None:
            self._load_data()
        
        return self._raw_data
        

class ThomsonReutersDataProvider:  
    def __init__(self,
                 start_date = dt.datetime(1990, 1, 1), 
                 end_date = dt.datetime(2015, 10, 20),
                 nbr_of_training_days=750, 
                 nbr_of_trading_days=250,
                 use_whitenoise_as_raw_data = False,
                 keep_dead_stocks_in_training = False):
        self.__start_date__ = start_date
        self.__end_date__ = end_date             
        self.__nbr_of_training_days__ = nbr_of_training_days
        self.__nbr_of_trading_days__ = nbr_of_trading_days
        self.__white_noise_as_raw_data__ = use_whitenoise_as_raw_data
        self.__batch_no__ = None
        self.__raw_data__ = None
        self.__SP500_const__ = None
        self.__keep_dead_stocks_in_training__ = keep_dead_stocks_in_training
    
    def get_raw_data_for_current_batch(self):
        if self.__batch_no__ is None:
            self.__batch_no__ = 0
            
        if self.__raw_data__ is None:
            raw_data, SP500_const = self.__read_data_from_disk__()
            self.__raw_data__ = raw_data
            self.__SP500_const__ = SP500_const

        raw_data = self.__raw_data__
        SP500_const = self.__SP500_const__            
        batch_no = self.__batch_no__
        

        # Determine data range for current batch        
        dates =  raw_data.index.drop_duplicates()
        
        
        try:
            
            first_date_of_training = dates[batch_no * self.__nbr_of_trading_days__]
            last_date_of_training = None
            last_date_of_training = dates[batch_no * self.__nbr_of_trading_days__\
                                          + self.__nbr_of_training_days__ -1]
            last_date_of_trading = dates[batch_no * self.__nbr_of_trading_days__\
                                         + self.__nbr_of_training_days__\
                                         + self.__nbr_of_trading_days__ - 1] 
        except:
            if last_date_of_training is not None:
                last_date_of_trading = dates[-1]
                if last_date_of_trading == last_date_of_training:
                    return None
            else:
                # no more data available
                return None

        print('[INFO] Returning raw data for batch', batch_no)

        
        if self.__keep_dead_stocks_in_training__ == False:
            # Obtain the S&P 500 constitution at the the last day of the training period
            stocks_in_SP500 = SP500_const.loc[last_date_of_training]
            stocks_in_SP500 = stocks_in_SP500.dropna()
            stocks_in_SP500 = stocks_in_SP500.index
        else:
            # Keep all stocks that have been part of the S&P 500 constituents for any time during the training period
            stocks_in_SP500 = SP500_const.loc[first_date_of_training:last_date_of_training]
            stocks_in_SP500 = stocks_in_SP500.dropna(axis='columns', how='all')
            stocks_in_SP500 = stocks_in_SP500.columns
        stocks_in_SP500 = stocks_in_SP500.intersection(raw_data.columns) # in case only closing_data for a small set of stocks will be used
    
        # Remove stocks that have not been part of the S&P 500 and filter date range   
        batch = raw_data[stocks_in_SP500].loc[first_date_of_training:last_date_of_trading]        
        batch = batch.dropna(axis='columns', how='all')     # In some cases the constituents matrix contains NaN stocks over the whole period      
        return batch


    def get_raw_data_for_next_batch(self):
        if self.__batch_no__ is None:
            self.__batch_no__ = 0
        else:
            self.__batch_no__ = self.__batch_no__ + 1          
        return self.get_raw_data_for_current_batch()
    
    
    def get_full_raw_data(self):
        self.get_raw_data_for_current_batch() # reads in data if required
        return self.__raw_data__
    
   
    def get_current_batch_no(self):
        return self.__batch_no__


    def reset_batch_no(self):
        self.__batch_no__ = 0

    def set_batch_no(self, batch_no):
        self.__batch_no__ = batch_no
        
        
    def __read_data_from_disk__(self):
    
        activateWhiteNoise = self.__white_noise_as_raw_data__    
    
        SP500_closing_data = pd.read_csv('../SP500Final/xtsSaniSP500.csv', index_col=0, parse_dates=True, sep=',', dayfirst=True)
        SP500_const = pd.read_csv('../SP500Final/xtsConstSP500.csv', index_col=0, parse_dates=True, sep=',', dayfirst=True)
            
        
        start = self.__start_date__ # dt.datetime(1990, 1, 1)         
        end = self.__end_date__     # dt.datetime(2015, 10, 30)
        
        # SP500_closing_data = pd.DataFrame(SP500_closing_data[:][['X905818', 'X912160']])   #, 
    
        print('[INFO] csv files read successfully')   
        SP500_closing_data = SP500_closing_data.loc[start:end]
        SP500_closing_data = SP500_closing_data.dropna(axis='columns', how='all')    # remove all stocks that have no prices at all  
        print('[INFO] Stocks not traded in training and test period removed - ', SP500_closing_data.shape[1], ' stocks remaining')
       
        SP500_index_data = pd.read_csv('../SP500Final/xtsIndexriSP500.csv', parse_dates=True, sep=';', decimal=',', dayfirst=True, index_col=0)
        SP500_index_data = SP500_index_data.loc[start:end]
        SP500_index_data.index.names=['Date']
        SP500_index_data.columns =  ['Close']
    
        
        if activateWhiteNoise:
            df = pd.DataFrame(np.random.normal(0, 0.0236, SP500_closing_data.shape).reshape(SP500_closing_data.shape))
            df.index = SP500_closing_data.index
            df.columns = SP500_closing_data.columns
            df = df+1
            df = df.cumprod()
            SP500_closing_data = df
            print('[INFO] White noise data activated')
        return SP500_closing_data, SP500_const



if __name__ == "__main__": 
    path_to_raw_data = '../Exchange/'
    dataprovider = CryptoMinuteDataProvider(path_to_raw_data)
    raw_data = dataprovider.get_full_raw_data()
    