import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
import shutil as shutil
import os as os    
import seaborn as sns # produces nicer charts plotted via pandas

class TrainTestSplitter:
    def __init__(self):
        self.__previous_last_date_of_training__ = None

    def split_data_into_train_and_test(self, features, targets, nbr_of_trading_days, len_seq, forecast_horizon):
        dates = targets.index.get_level_values('Date').drop_duplicates()
        first_date_of_training = dates[0]
        
                
        SAFETY_BUFFER = 5
        if self.__previous_last_date_of_training__ is None:
            last_date_of_training = dates[-nbr_of_trading_days - (forecast_horizon+1 + SAFETY_BUFFER) - 1]
        else:
            last_date_of_training = dates[np.where(dates == self.__previous_last_date_of_training__)[0][0] + nbr_of_trading_days]
            remaining_nbr_of_trading_days = len(dates) - np.where(dates == last_date_of_training)[0][0] - 1 - (forecast_horizon + 1)
            nbr_of_trading_days = min(nbr_of_trading_days, remaining_nbr_of_trading_days)
        
        self.__previous_last_date_of_training__ = last_date_of_training        
        
        if len_seq == 0:   # no lstm
            first_date_of_test = dates[-(nbr_of_trading_days)]  
        else:              # train and test set need to overlap by len_seq-1 days
            first_date_of_test = dates[-(nbr_of_trading_days + len_seq - 1)]   

        
        
        
        
        training_features = features.loc[first_date_of_training:last_date_of_training]
        training_targets = targets.loc[first_date_of_training:last_date_of_training]
        
        test_features = features.loc[first_date_of_test:]
        test_targets = targets.loc[first_date_of_test:]
        
        
        print('[INFO] Data split into train and test set:')
        print('[INFO] Training timeframe:', first_date_of_training.date(), '-', 
              last_date_of_training.date())
        print('[INFO] Test timeframe:', first_date_of_test.date(), '-', 
              dates[-1].date())
        print('[INFO] First date predicted:', dates[-(nbr_of_trading_days)].date(), '\n')
        
        return training_features, test_features, training_targets, test_targets
    

def discretize_values(return_data, nbr_of_bins=2): 
    return_data = return_data.dropna()
    if nbr_of_bins != 2:
        assert(False, 'Not implemented yet!')
    
    daily_median = return_data.groupby(level='Date').median()
    daily_median.columns = ['daily_median']
    result = return_data.join(daily_median)
    result['binary_target'] = 0
    result.loc[result[return_data.columns[0]] >= result.daily_median, 'binary_target'] = 1
    result = pd.DataFrame(result.binary_target)
    result.columns = ['return_to_predict']
    return result
    
    
#    print('[INFO] Discretizing actual values with ' + str(nbr_of_bins) + ' bins')    
#    return_data = pd.DataFrame(return_data)
#    dates = return_data.index.get_level_values(level='Date').drop_duplicates()
#
#    result = []    
#    for a_date in dates:
#        df = return_data.ix[return_data.index.get_level_values(level='Date') == a_date]
#        try:
#            df = df.apply(lambda x : pd.qcut(x, nbr_of_bins, np.arange(0, nbr_of_bins)))       
#        except:
#            print('[DEBUG] Couldn\'t discretize returns for date ', a_date)                        
#            df = df.apply(lambda x : pd.qcut(x, 1, labels=[np.int(nbr_of_bins / 2)]))
#        result.append(df) 
#        
#    for df in result:
#        df.return_to_predict = pd.to_numeric(df.return_to_predict)
#    result = pd.concat(result, axis=0)
#    result = pd.DataFrame(result)
#    result.columns=['return_to_predict']
    return result
   


def backtest_model(prediction_probas, k, dataprovider, forecast_horizon):
    """ k denotes the top-k and flop-k stocks that are traded based on the 
        final probability ranking    
    """
    raw_data = dataprovider.get_full_raw_data()
    actual_returns = raw_data/raw_data.shift(forecast_horizon) - 1
    actual_returns = actual_returns.stack()
    actual_returns.index.names = ['Date', 'Stock']
    actual_returns = pd.DataFrame(actual_returns)
    actual_returns.columns = ['actual_return']
    
    
    top_k = prediction_probas['probability'].groupby(level='Date').nlargest(k)
    top_k.index = top_k.index.droplevel(0)
    top_k = pd.DataFrame(top_k)
    top_k = top_k.join(actual_returns)
    top_k.columns = ['probability', 'return_long']
    top_k = top_k.groupby(level='Date').mean()
    
    flop_k = prediction_probas['probability'].groupby(level='Date').nsmallest(k)
    flop_k.index = flop_k.index.droplevel(0)
    flop_k = pd.DataFrame(flop_k)
    flop_k = flop_k.join(actual_returns)
    flop_k.columns = ['probability', 'return_short']
    flop_k['return_short'] = flop_k['return_short'] * (-1)
    flop_k = flop_k.groupby(level='Date').mean()
    del(flop_k['probability'])    
    

    total_returns = top_k.join(flop_k)    
    del(total_returns['probability'])    
    total_returns['return_total'] = total_returns['return_long'] + total_returns['return_short']
    
    
    total_returns = total_returns/forecast_horizon
    print('[INFO] Results of backtesting\n', total_returns.describe())  
    
    return total_returns
    
    

def evaluate_model_performance(targets, prediction_probas):
    accuracy = accuracy_score(targets, np.round(prediction_probas, 0))
    log_loss_score = log_loss(targets, prediction_probas)

    print('[INFO] accuracy:', accuracy, '\nlog_loss:', log_loss_score)
    return accuracy, log_loss_score


    
def compile_performance_summary(all_predictions, all_realized_returns, dataprovider, forecast_horizon, k):
    print('[INFO] Compiling results')
    PATH_TO_SUMMARY = './performance_summary'
    
    all_realized_returns_excluding_tc = all_realized_returns # tc = transaction costs
    all_realized_returns = all_realized_returns.copy()
    
    """ (1) Determine true targets """
    raw_data = dataprovider.get_full_raw_data()
    true_targets = raw_data/raw_data.shift(forecast_horizon) - 1
    true_targets = true_targets.stack()
    true_targets.index.names = ['Date', 'Stock']
    true_targets = pd.DataFrame(true_targets)
    true_targets.columns = ['return_to_predict']
    true_targets = true_targets.join(all_predictions)
    true_targets = true_targets.dropna()
    true_targets = pd.DataFrame(true_targets['return_to_predict'])
    true_targets = discretize_values(true_targets)
    true_targets.columns = ['target']
    
    """ (2) Determine accuracy and log_loss for all stocks """
    evaluation_df = all_predictions.join(true_targets)
    evaluation_df['accuracy'] = (np.round(evaluation_df.probability, 0) == evaluation_df.target) * 1
    evaluation_df['log_loss'] = (evaluation_df.target * np.log(evaluation_df.probability) + \
                                 (1 - evaluation_df.target) * np.log(1 - evaluation_df.probability)) * (-1)
    
    overall_accuracy = evaluation_df['accuracy'].mean()
    overall_log_loss = evaluation_df['log_loss'].mean()        
                             
    accuracy_by_year = evaluation_df['accuracy'].groupby(level='Date').mean().groupby(lambda x : x.year).mean()
    accuracy_by_year = pd.DataFrame(accuracy_by_year)
    log_loss_by_year = evaluation_df['log_loss'].groupby(level='Date').mean().groupby(lambda x : x.year).mean()
    log_loss_by_year = pd.DataFrame(log_loss_by_year)
    
    """ (3) Determine accuracy and log_loss for the top/flop k stocks that are traded """
    top_k = evaluation_df['probability'].groupby(level='Date').nlargest(k)
    flop_k = evaluation_df['probability'].groupby(level='Date').nsmallest(k)
    top_flop_k = pd.concat([top_k, flop_k], axis=0)
    top_flop_k.index = top_flop_k.index.droplevel(0)    
    top_flop_k = pd.DataFrame(top_flop_k)
    top_flop_k = top_flop_k.join(evaluation_df[['accuracy', 'log_loss']])
    
    overall_accuracy_top_flop_k = top_flop_k['accuracy'].mean()
    overall_log_loss_top_flop_k = top_flop_k['log_loss'].mean()        
    
    accuracy_by_year_top_flop_k = top_flop_k['accuracy'].groupby(level='Date').mean().groupby(lambda x : x.year).mean() 
    accuracy_by_year_top_flop_k = pd.DataFrame(accuracy_by_year_top_flop_k)
    log_loss_by_year_top_flop_k = top_flop_k['log_loss'].groupby(level='Date').mean().groupby(lambda x : x.year).mean() 
    log_loss_by_year_top_flop_k = pd.DataFrame(log_loss_by_year_top_flop_k)
    
    all_realized_returns['return_long'] = all_realized_returns['return_long'] - 0.001/forecast_horizon
    all_realized_returns['return_short'] = all_realized_returns['return_short'] - 0.001/forecast_horizon
    all_realized_returns['return_total'] = all_realized_returns['return_total'] - 0.002/forecast_horizon
    

    """ (4) Open, fill and store html template """
    
    template = open('ml_testbench_performance_summary_clean.html').read()
    
    FORMAT = "{:10.4f}"
    
    template = template.replace('{{=overall_accuracy}}', FORMAT.format(overall_accuracy))
    template = template.replace('{{=overall_log_loss}}', FORMAT.format(overall_log_loss))
    
    template = template.replace('{{=overall_accuracy_top_flop_k}}', FORMAT.format(overall_accuracy_top_flop_k))
    template = template.replace('{{=overall_log_loss_top_flop_k}}', FORMAT.format(overall_log_loss_top_flop_k))
    
    template = template.replace('{{=return_long}}', FORMAT.format(all_realized_returns['return_long'].mean()))
    template = template.replace('{{=return_short}}', FORMAT.format(all_realized_returns['return_short'].mean()))
    template = template.replace('{{=return_total}}', FORMAT.format(all_realized_returns['return_total'].mean()))

        
    template = template.replace('{{=return_summary_statistics}}', all_realized_returns.describe().to_html())
    return_summary_by_year = all_realized_returns.groupby(lambda x : x.year).mean()
    
    template = template.replace('{{=return_summary_by_year}}', return_summary_by_year.to_html())

    template = template.replace('{{=accuracy_summary_by_year}}', accuracy_by_year.to_html())
    template = template.replace('{{=accuracy_summary_by_year_top_flop_k}}', accuracy_by_year_top_flop_k.to_html())
    
    template = template.replace('{{=log_loss_summary_by_year}}', log_loss_by_year.to_html())
    template = template.replace('{{=log_loss_summary_by_year_top_flop_k}}', log_loss_by_year_top_flop_k.to_html())
    
    
    
    try:
        shutil.rmtree(PATH_TO_SUMMARY)
    except:
        print('[INFO] Coudn\'t delete summary folder')
    
    try:
        os.makedirs(PATH_TO_SUMMARY)
        chart_total_returns = 'chart_total_returns.png'
        (all_realized_returns/forecast_horizon + 1).cumprod().plot(figsize=(10,7)).get_figure().savefig(PATH_TO_SUMMARY + '/' + chart_total_returns)
        template = template.replace('{{=chart_total_returns}}', chart_total_returns)
        
        chart_returns_by_year = 'chart_returns_by_year.png'
        (all_realized_returns).groupby(lambda x : x.year).mean().plot(kind='bar', figsize=(10,3)).get_figure().savefig(PATH_TO_SUMMARY + '/' + chart_returns_by_year)
        template = template.replace('{{=chart_returns_by_year}}', chart_returns_by_year)
            
        chart_accuracy_by_year = 'chart_accuracy_by_year.png'        
        accuracy_by_year.plot(kind='bar', title='Accuracy by year (all stocks)', 
                              ylim=(accuracy_by_year['accuracy'].min()*0.98, 
                                    accuracy_by_year['accuracy'].max()*1.02), figsize=(10,3)).get_figure().savefig(PATH_TO_SUMMARY + '/' + chart_accuracy_by_year)
        template = template.replace('{{=chart_accuracy_by_year}}', chart_accuracy_by_year)
        


        chart_accuracy_by_year_top_flop_k = 'chart_accuracy_by_year_top_flop_k.png'        
        accuracy_by_year_top_flop_k.plot(kind='bar', title='Accuracy by year (top/flop-k stocks)',
                                         ylim=(accuracy_by_year_top_flop_k['accuracy'].min()*0.98, 
                                               accuracy_by_year_top_flop_k['accuracy'].max()*1.02), figsize=(10,3)).get_figure().savefig(PATH_TO_SUMMARY + '/' + chart_accuracy_by_year_top_flop_k)
        template = template.replace('{{=chart_accuracy_by_year_top_flop_k}}', chart_accuracy_by_year_top_flop_k)



        chart_log_loss_by_year = 'chart_log_loss_by_year.png'        
        log_loss_by_year.plot(kind='bar', title='Log_loss by year (all stocks)', 
                              ylim=(log_loss_by_year['log_loss'].min()*0.98, 
                                    log_loss_by_year['log_loss'].max()*1.02), figsize=(10,3)).get_figure().savefig(PATH_TO_SUMMARY + '/' + chart_log_loss_by_year)
        template = template.replace('{{=chart_log_loss_by_year}}', chart_log_loss_by_year)
        


        chart_log_loss_by_year_top_flop_k = 'chart_log_loss_by_year_top_flop_k.png'        
        log_loss_by_year_top_flop_k.plot(kind='bar', title='Log_loss by year (top/flop-k stocks)',
                                         ylim=(log_loss_by_year_top_flop_k['log_loss'].min()*0.98, 
                                               log_loss_by_year_top_flop_k['log_loss'].max()*1.02), figsize=(10,3)).get_figure().savefig(PATH_TO_SUMMARY + '/' + chart_log_loss_by_year_top_flop_k)
        template = template.replace('{{=chart_log_loss_by_year_top_flop_k}}', chart_log_loss_by_year_top_flop_k)



        
        
        summary = open(PATH_TO_SUMMARY + '/performance_summary.html', "w")    
        summary.write(template)
        summary.close() 
        all_predictions.to_csv(PATH_TO_SUMMARY + '/all_predictions.csv')
        all_realized_returns_excluding_tc.to_csv(PATH_TO_SUMMARY + '/all_realized_returns_exkl_tc.csv')
        true_targets.to_csv(PATH_TO_SUMMARY + '/true_targets.csv')
    
    except:
        print('[ERROR] Couldn\'t write performance summary to disk')


def store_predictions_of_batch(predictions, batch_no):
    PATH_TO_PREDICTIONS = './predictions'
    if os.path.isdir(PATH_TO_PREDICTIONS) == False:
        os.makedirs(PATH_TO_PREDICTIONS)
    
    predictions.to_csv(PATH_TO_PREDICTIONS + '/prediction_probas_batch_' + str(batch_no) + '.csv')
    print('[INFO] Predictions of batch', batch_no, 'stored on disk in folder', PATH_TO_PREDICTIONS)
    
