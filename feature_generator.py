#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 15:15:37 2017

@author: thomas
"""


import pandas as pd
import helper as helper


""" ----------------------------------------------------------------------- """
""" Some functions to compute features                                      """
""" ----------------------------------------------------------------------- """



def calculate_discrete_multiperiod_returns(price_df, period):
    result = price_df/price_df.shift(period) - 1
    result = result.stack(dropna=False)
    result.index.names=['Date', 'Stock']
    result = pd.DataFrame(result, columns=['return_t_t-' + str(period)])
    return result

    
def calculate_momentum_rank(price_df, period):
    result = price_df/price_df.shift(period) - 1; 
    result = result.stack(dropna=False)
    result.index.names=['Date', 'Stock']
    result = pd.DataFrame(result, columns=['multi_period_return'])
    result['mom_t_' + str(period)] = result.groupby(level='Date').rank() / \
                                     result.groupby(level='Date').count()
    del(result['multi_period_return'])
    return result


def calculate_rolling_mean(price_df, window):
    result = (price_df/price_df.shift(1)-1)
    result = result.rolling(window=window).mean()
    result = result.stack(dropna=False)        
    result.index.names = ['Date', 'Stock']
    result = pd.DataFrame(result)
    result.columns = ['rolling_mean_' + str(window) + '_days']
    return result     


def calculate_rolling_std(price_df, window):
    result = (price_df/price_df.shift(1)-1)
    result = result.rolling(window=window).std()
    result = result.stack(dropna=False)        
    result.index.names = ['Date', 'Stock']
    result = pd.DataFrame(result)
    result.columns = ['rolling_std_' + str(window) + '_days']
    return result   
        

""" ----------------------------------------------------------------------- """
""" Main function to compile features and targets                           """
""" ----------------------------------------------------------------------- """

def compile_features_and_targets(raw_data, forecast_horizon, nbr_of_targets, feature_generation_functions_and_arguments):
    print('[INFO] Compiling features and targtes')
    raw_data = raw_data.close.unstack()
    targets = raw_data/raw_data.shift(forecast_horizon) - 1
    targets = targets.stack(dropna = False)
    targets = pd.DataFrame(targets)
    targets.columns = ['return_to_predict']
    targets.index.names = ['Date', 'Stock']
    
    raw_data = raw_data.shift(forecast_horizon + 1) # shift to eliminate look-ahead bias
    
    
    features = None
    for a_tuple in feature_generation_functions_and_arguments:
        print('[INFO] Building feature', a_tuple)
        function_to_call = a_tuple[0]
        function_arguments = a_tuple[1]
        
        if features is None:
            features = function_to_call(raw_data, **function_arguments)
        else:
            features = features.join(function_to_call(raw_data, **function_arguments))

    
    # join features and targets to drop NA's where necessary
    merged = targets.join(features)
    merged = merged.dropna()
    
    targets = pd.DataFrame(merged['return_to_predict'])
    targets = helper.discretize_values(targets, nbr_of_bins = nbr_of_targets)
    
    features = merged
    del(features['return_to_predict'])
    
    return features, targets

