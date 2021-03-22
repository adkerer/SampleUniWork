#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 10:37:23 2019

@author: AdamEriksson
"""


from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
#from pandas.tools.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests 
from sklearn.metrics import mean_absolute_error
#import csv

def parser(x):
    return datetime.strptime(x,'%Y-%m-%d %H:%M:%S')
def plot_series():
    print("hej")
    series = read_csv("june_production.csv",header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    #series = series[0:3500]
    correlation_data = series[0:100]
    #start with looking at three days
    print(series.head())
    #prod = series.ix[0,:]
    #print(prod)
    #series.plot()
    autocorrelation_plot(correlation_data)
    #plot_acf(prod, ax=pyplot.gca())
    pyplot.show()
    
def arima_residuals():
    series = read_csv("june_production_random_shuffle.csv",header=0, parse_dates=[1], index_col=1, squeeze=True, date_parser=parser)
    model = ARIMA(series, order=(3,1,0))
    model_fit = model.fit(disp=0)
    # print(model_fit.summary())
    residuals = DataFrame(model_fit.resid)
    residuals.plot()
    pyplot.show()
    residuals.plot(kind='kde')
    pyplot.show()
    print(residuals.describe())
    print(type(residuals))
    residuals.to_csv("residual_test")
    
def arima_train(p=3,d=1,q=0):
    series = read_csv("june_production_random_shuffle.csv",header=0, parse_dates=[1], index_col=1, squeeze=True, date_parser=parser)
    #series = series[0:3500]
    print(series.head())
    X = series.values
    #print(X[0:10])
    #size = int(len(X) * 0.66
    #train, test = X[0:size], X[size:len(X)]
    last_five_days = 5842
    train, test = X[0:len(X)-last_five_days] , X[len(X)-last_five_days:len(X)]
    history = [x for x in train]
    predictions = list()
    persistance_prediction = list()
    persistance_prediction.append(test[0])
    persistance_prediction.extend(test[0:len(test)-1])
    for t in range(len(test)):
        model = ARIMA(history, order=(p,d,q))
        model_fit = model.fit(disp=0)
        #import pdb; pdb.set_trace()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        #persistance_prediction.append[obs]
        print('predicted=%f, expected=%f' % (yhat, obs))
    
    
    rms_error = sqrt(mean_squared_error(test, predictions))
    rms_error_persistance = sqrt(mean_squared_error(test, persistance_prediction))
    print('Test RMS: %.3f' % rms_error)
    print('Persistance RMS: %.3f' % rms_error_persistance)
    print(model_fit.summary())
    # plot
    pyplot.plot(test)
    pyplot.plot(predictions, color='red')
    pyplot.show()
    residual_filename = "residuals_" + str(p) + "_" + str(d) + "_" + str(q) + "_030518"
    residual_summary(model_fit,residual_filename)

def arima_train_diff():
    series = read_csv("june_production.csv",header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    #series = series[0:3500]
    Y = series.values
    Y_create_diff = [0]
    Y_create_diff.extend(Y[0:len(Y)-1])
    X = list(np.array(Y)-np.array(Y_create_diff))
    #size = int(len(X) * 0.66
    #train, test = X[0:size], X[size:len(X)]
    oneday = 1200
    train, test = X[0:len(X)-oneday] , X[len(X)-oneday:len(X)]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=(3,1,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
    rms_error = sqrt(mean_squared_error(test, predictions))
    print('Test RMS: %.3f' % rms_error)
    # plot
    pyplot.plot(test)
    pyplot.plot(predictions, color='red')
    pyplot.show()

def plot_residuals(series):
    plt.figure(1, (20,10))
    
    plt.subplot(121)
    plt.title('Autocorrelation plot residuals')
    autocorrelation_plot(series)
    
    plt.subplot(122)
    #plt.title('Production Week in June [W]')
    n, bins, patches = pyplot.hist(bins='auto')
    
'''def list_to_csv(filename,data):
    with open(filename, mode='w') as file:
        file_writer = csv.writer(file, delimiter=',')
        
        for i in range(len(data)):
            file_writer.writerow(data[i])
'''    
def residual_summary(model_fit, filename):
    # print(model_fit.summary())
    residuals = DataFrame(model_fit.resid)
    residuals.plot()
    pyplot.show()
    residuals.plot(kind='kde')
    pyplot.show()
    print(residuals.describe())
    residuals.to_csv(filename)
    
def arima_june(p=3,d=1,q=0):
    series = read_csv("june_production_random_shuffle_percent_30_min_interval.csv",header=0, parse_dates=[1], index_col=1, squeeze=True, date_parser=parser)
    #print(series.head())
    #X = to_percent(series.values)
    #print(X[0:50])
    #print(series.values[0:50])
    #Y = DataFrame(X)
    #series.replace(series.values, Y)
    #print(series[0:100])
    #series.to_csv("june_production_random_shuffle_percent.csv")
    last_five_days = int(0.2 * len(series))
    train, test = series[0:len(series)-last_five_days] , series[len(series)-last_five_days:len(series)]
    history = [x for x in train]
    predictions = list()
    persistence_prediction = list()
    persistence_prediction.append(test[0])
    persistence_prediction.extend(test[0:len(test)-1])
    for t in range(len(test)):
        model = ARIMA(history, order=(p,d,q))
        model_fit = model.fit(disp=0)
        #import pdb; pdb.set_trace()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        #persistance_prediction.append[obs]
        print('predicted=%f, expected=%f' % (yhat, obs))
    rms_error = sqrt(mean_squared_error(test, predictions))
    rms_error_persistence = sqrt(mean_squared_error(test, persistence_prediction))
    print('Test RMS: %.3f' % rms_error)
    print('Persistance RMS: %.3f' % rms_error_persistence)
    print(model_fit.summary())
    # plot
    pyplot.plot(test.values)
    pyplot.plot(predictions, color='red')
    pyplot.show()
    residual_filename = "30min_residuals_" + str(p) + "_" + str(d) + "_" + str(q) + "_050519.csv"
    residual_summary(model_fit,residual_filename)
    
    
def granger(minute_interval = 1):
    filename_prod = "june_july_august_" + str(minute_interval)+ "_final_production.csv"
    filename_wind = "june_july_august_" + str(minute_interval)+ "_final_wind.csv"
    filename_hour = "june_july_august_" + str(minute_interval)+ "_final_hours.csv"
    prod_series = read_csv(filename_prod,header = None)
    wind_series = read_csv(filename_wind,header = None) 
    hour_series = read_csv(filename_hour,header = None)
    #list_of_series = [prod_series[0].tolist(), wind_series[0].tolist(), wind_series[1].tolist()]
    frame = { 'production': prod_series[0].diff().dropna(), 'wind': wind_series[0].diff().dropna(), 'hour': hour_series[0].diff().dropna()} #'timestamp':wind_series[1] 
    df = pd.DataFrame(frame)
    print(df.head())
    print(grangercausalitytests(df[['production','wind']],10))
    
def arima_run(p,d,q, minute_interval = 1):
    filename = "july_" + str(minute_interval)+ "_production_rsn.csv"
    prod_series = read_csv(filename,header = None)
    print(prod_series.head(10))
    values = prod_series[0].tolist()
    #history 24 days. Forecast on last 4 
    points_per_day = int(1440/minute_interval)
    history_length = 24*points_per_day #june and july
    test_length = 4 * points_per_day
    history = values[0:history_length]
    test = values[history_length:history_length+test_length]
    print(test[0:10])
    #print(test.iloc[0].toList())
    
    predictions = list()
    persistence_prediction = list()
    persistence_prediction.append(test[0])
    persistence_prediction.extend(test[0:len(test)-1])
    
    #model_nontest = ARIMA(history, order= (p,d,q))
    #model_nontest_fit = model_nontest.fit(disp=0)
    
    for t in range(len(test)):
        model = ARIMA(history, order=(p,d,q))
        model_fit = model.fit(disp=0)
        #import pdb; pdb.set_trace()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        #persistance_prediction.append[obs]
        print('predicted=%f, expected=%f' % (yhat, obs))
    
    rms_error = sqrt(mean_squared_error(test, predictions))
    rms_error_persistence = sqrt(mean_squared_error(test, persistence_prediction))
    print('Test RMS: %.4f' % rms_error)
    print('Persistance RMS: %.4f' % rms_error_persistence)
    print(model_fit.summary())
    #print(model_nontest_fit.summary())
    frame_predictions_test = { 'predictions': predictions, 'test': test}
    df_predictions_test = pd.DataFrame(frame_predictions_test)
    #filename = "predictions_" + str(p) + "_" + str(d) + "_" + str(q) + "_arima.csv"
    filename = "predictions_" + str(p) + "_" + str(d) + "_" + str(q) + "_july_arima.csv"
    df_predictions_test.to_csv(filename)
    

    #(5,1,1,10) stor skillnad mellan exog och icke exog
def arimax_run(p,d,q, minute_interval = 1):
    filename_prod = "july_" + str(minute_interval)+ "_production_rsn.csv"
    filename_wind = "july_" + str(minute_interval)+ "_wind_rsn.csv"
    prod_series = read_csv(filename_prod,header = None)
    wind_series = read_csv(filename_wind,header = None)
    points_per_day = int(1440/minute_interval)
    history_length = 24*points_per_day #first 24 days of july

    frame = { 'production': prod_series[0].iloc[0:history_length], 'wind': wind_series[0].iloc[0:history_length],'timestamps':prod_series[1].iloc[0:history_length]}
    df = pd.DataFrame(frame)
    print(df.head())
    #history june, july. Train: 4 days. Check MAE on 1 hour, 1 day and 10 days
    
    test_length = 4 * points_per_day
    #print(test[0:10])
    #print(test.iloc[0].toList())
    
    predictions = list()
    test = list()
    persistence_prediction = list()
    persistence_prediction.append(prod_series[0].iloc[history_length].tolist())
    persistence_prediction.extend(prod_series[0].iloc[history_length:history_length+test_length-1].tolist())
    #print(len(persistence_prediction))
    
    #model_nontest = ARIMA(endog = df['production'], order= [p,d,q], exog = df[['wind']])
    #model_nontest_fit = model_nontest.fit(disp=0)

    for t in range(test_length):
        i = t + history_length 
        new_values = [prod_series[0].iloc[i],wind_series[0].iloc[i],prod_series[1].iloc[i]]
        
        model = ARIMA(endog = df['production'], exog = df[['wind']], order= [p,d,q])
        model_fit = model.fit(disp=0)
        #print(new_values[1])
        exog_next_timestep = np.array([new_values[1]])
        output = model_fit.forecast(steps = 1, exog = exog_next_timestep)
        yhat = output[0]
        predictions.append(yhat)
        test.append(new_values[0])
        df.loc[i]  = new_values
        #create exog variable same length as prod. then add the exog variable before forecasting...

        print('predicted=%f, expected=%f' % (yhat, new_values[0]))
   
    
    rms_error = sqrt(mean_squared_error(test, predictions))
    rms_error_persistence = sqrt(mean_squared_error(test, persistence_prediction))
    print('Test RMS: %.4f' % rms_error)
    print('Persistance RMS: %.4f' % rms_error_persistence)
    print(model_fit.summary())
   
    #print(model_nontest_fit.summary())
    frame_predictions_test = { 'predictions': predictions, 'test': test}
    df_predictions_test = pd.DataFrame(frame_predictions_test)
    #filename = "predictions_" + str(p) + "_" + str(d) + "_" + str(q) + "_arimax.csv"
    filename = "predictions_" + str(p) + "_" + str(d) + "_" + str(q) + "_july_arima.csv"
    df_predictions_test.to_csv(filename)

def arimax_predict(p,d,q, hours_to_predict, minute_interval = 1):
    filename_prod = "june_july_august_" + str(minute_interval)+ "_final_production.csv"
    filename_wind = "june_july_august_" + str(minute_interval)+ "_final_wind.csv"
    filename_hour = "june_july_august_" + str(minute_interval)+ "_final_hours.csv"
    prod_series = read_csv(filename_prod,header = None)
    wind_series = read_csv(filename_wind,header = None)
    hour_series = read_csv(filename_hour,header = None)
    points_per_day = int(1440/minute_interval)
    steps_predict = int(hours_to_predict * points_per_day/24)
    
    history_length = int((29+28+3.5)*points_per_day) #june and july and four days in aug
    
    #frame = { 'production': prod_series[0].iloc[0:history_length], 'wind': wind_series[0].iloc[0:history_length],'timestamps':prod_series[1].iloc[0:history_length]}
    start =  int((28+1.5)*points_per_day)
    frame = { 'production': prod_series[0].iloc[start:history_length], 'wind': wind_series[0].iloc[start:history_length],'hour':hour_series[0].iloc[start:history_length],'timestamps':prod_series[1].iloc[start:history_length]}
    df = pd.DataFrame(frame)
    print(df.head())
    #history june, july. Train: 4 days. Check MAE on 1 hour, 1 day and 10 days
    
    #print(test[0:10])
    #print(test.iloc[0].toList())
    
    predictions_x = list()
    predictions = list()
    #wind_values = np.array(wind_series[0].iloc[history_length:history_length+steps_predict].tolist())
    hour_values = np.array(hour_series[0].iloc[history_length:history_length+steps_predict].tolist())
    
    '''model_x = ARIMA(endog = df['production'], exog = df[['wind']], order= [p,d,q])
    model_x_fit = model_x.fit(disp=0)
    print(model_x_fit.summary())
    output_x, se_x, conf_x = model_x_fit.forecast(steps = steps_predict, exog = wind_values)
   ''' 
   
    model_x = ARIMA(endog = df['production'], exog = df[['hour']], order= [p,d,q])
    model_x_fit = model_x.fit(disp=0)
    print(model_x_fit.summary())
    output_x, se_x, conf_x = model_x_fit.forecast(steps = steps_predict, exog = hour_values)
    
    model = ARIMA(endog = df['production'], order= [p,d,q])
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    output, se, conf = model_fit.forecast(steps = steps_predict)
    
    predictions_x.extend(output_x)
    predictions.extend(output)
    actual_values =  prod_series[0].iloc[history_length:history_length+steps_predict].tolist()
    
    print(type(output))
    print(len(predictions_x),len(predictions), len(actual_values))
    mae_x = mean_absolute_error(actual_values,predictions_x)
    mae = mean_absolute_error(actual_values,predictions)
    print(len(output),len(se),len(conf))
    
    #print(model_x_fit.summary())
    #print(model_fit.summary())
    print("Mean Absolute Error. ARIMAX:%s, ARIMA:%s" %(mae_x, mae))
    
    frame_end = {'actual_values':actual_values, 'output': output.tolist(), 'se': se.tolist(), 'conf':conf.tolist(), 'output_x': output_x.tolist(), 'se_x': se_x.tolist(), 'conf_x':conf_x.tolist()}
    df = pd.DataFrame(frame_end)
    filename = 'multistep_prediction_' + str(p) + '_' + str(d) + '_' + str(q) + '_' + str(hours_to_predict) + '_hours_' + str(minute_interval) + '_min.csv'
    df.to_csv(filename)
    
def convert_to_float_list(list1, listx):
    #conf, conf_x = [], []
    lower, lower_x, upper, upper_x = [],[],[],[]
    for i in range(len(list1)):
        str_list = list1[i].replace('[','').replace(']','').split(', ')
        str_list_x = listx[i].replace('[','').replace(']','').split(', ')
        
        #float_list = [float(str_list[0]),float(str_list[1])]
        #float_list_x = [float(str_list_x[0]),float(str_list_x[1])]
        lower.append(float(str_list[0]))
        upper.append(float(str_list[1]))
        lower_x.append(float(str_list_x[0]))
        upper_x.append(float(str_list_x[1]))
        
        #conf.append(float_list)
        #conf_x.append(float_list_x)
        
        #conf.append(float(str_list[0]),float(str_list[1]))
        #conf_x.append(float(str_list_x[0]),float(str_list_x[1]))
    return lower, lower_x, upper, upper_x
        
def forecast_comparison(p,d,q, hours_to_predict, minute_interval):
    filename = 'multistep_prediction_' + str(p) + '_' + str(d) + '_' + str(q) + '_' + str(hours_to_predict) + '_hours_' + str(minute_interval) + '_min.csv'
    filename_prod = "june_july_august_" + str(minute_interval)+ "_final_production.csv"
    prod_series = read_csv(filename_prod,header = None)
    points_per_day = int(1440/minute_interval)
    history_start = int((29+28+1.375)*points_per_day)
    history_end = int((29+28+1.5)*points_per_day)
    steps_predict = int(hours_to_predict * points_per_day/24)
    
    index_array = range(history_end-history_start-1,history_end-history_start -1 + steps_predict)
    production = prod_series[0].iloc[history_start:history_end].tolist() #.reindex(range(0,history_end-history_start))
    history_series = pd.Series(production)
    print(range(history_end-history_start,steps_predict))
    df = pd.read_csv(filename)
    #print(production_series.head(-1))
    print(df['se_x'].head())

    lower, lower_x, upper, upper_x = convert_to_float_list(df['conf'], df['conf_x'])
    #se, se_x = df['se'].tolist(), df['se_x'].tolist()
    output, output_x = df['output'].tolist(), df['output_x'].tolist()
    actual_values = df['actual_values'].tolist()
    
    #print(conf[:,0])
    #se_series, se_x_series = pd.Series(se,index = index_array), pd.Series(se_x,index = index_array)
    output_series, output_x_series  = pd.Series(output,index = index_array), pd.Series(output_x,index = index_array)
    actual_values_series = pd.Series(actual_values, index = index_array)
    lower_series, lower_series_x = pd.Series(lower, index=index_array), pd.Series(lower_x, index=index_array)
    upper_series, upper_series_x = pd.Series(upper, index=index_array), pd.Series(upper_x, index=index_array)
    
    plt.figure(1, (20,10))
    plt.subplot(2,1,1)
    plt.plot(history_series, label='history')
    plt.plot(actual_values_series, label='actual')
    plt.plot(output_series, label='ARIMA forecast')
    plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
    plt.legend(loc='upper left')
    plt.title('A')
    
    plt.legend(loc='upper left')
    plt.subplot(2,1,2)
    plt.plot(history_series, label='history')
    plt.plot(actual_values_series, label='actual')
    plt.plot(output_x_series, label='ARIMAX forecast')
    plt.fill_between(lower_series_x.index, lower_series_x, upper_series_x, 
                 color='k', alpha=.15)
    plt.title('B')
    plt.legend(loc='upper left')
    
    
    
    
    