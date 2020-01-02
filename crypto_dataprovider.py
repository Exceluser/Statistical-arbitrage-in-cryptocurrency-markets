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