# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 21:50:56 2018

@author: adein
"""

import numpy as np
np.random.seed(1)
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
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
model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
model.fit(training_features, training_targets)


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



