# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:48:14 2019

@author: adame
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

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared, DotProduct
    

def plot_days(minute_interval = 10):
    filename = "june_july_august_" + str(minute_interval)+ "_final_production.csv"
    prod_df = read_csv(filename,header = None)
    prod_list = prod_df[0].tolist()
    timestamp_list = prod_df[1].tolist()
    print(len(prod_list))
    days = [[] for i in range(61)]
    values_per_day = int(1440/minute_interval)
    days_look_at = [[] for i in range(8)]
    timestamps = [[] for i in range(8)]
    counter = 0
    for i in range(len(days)):
        for j in range(values_per_day):
            days[i].append(prod_list[i*values_per_day+j])
            if(i in range(30,38)):
                timestamps[counter].append(timestamp_list[i*values_per_day+j])
        if(i < 31):
            plt.figure(1, (20,10))
            plt.subplot(6,6,i+1)
            plt.plot(days[i])
            plt.title("Indx:%s" % i)
        else:
            plt.figure(2, (20,10))
            plt.subplot(6,5,i-30)
            plt.plot(days[i])
            plt.title("Index:%s" % i)
        #print(len(days[i]))
        plt.figure(3, (20,10))
        if(i in range(30,38)):
            days_look_at[counter].extend(days[i])
            plt.subplot(1,8,counter+1)
            plt.plot(days[i])
            plt.title("Day:%s" % (counter+1))
            print(len(days_look_at[counter]),len(timestamps[counter]))
            counter += 1
            print(prod_df[1].iloc[i*values_per_day])

    for i in range(len(timestamps)):
        for j in range(values_per_day):
            if(j>2):
                if(days_look_at[i][j-2] == 0 and days_look_at[i][j-1] == 0 and days_look_at[i][j] > 0):
                    print(timestamps[i][j])
            if((values_per_day-3)>j):
                if (days_look_at[i][j+2] == 0 and days_look_at[i][j+1] == 0 and days_look_at[i][j] > 0):
                    print(timestamps[i][j])
    
    #history: index 30 to 36, predict on index 37
    #days 07-18, 07-09, 07-15,07-01, 07-11, 07-04, 07-10, test 07-02
    #do length between 1:30 and 20:30, 114 points per day
    cut_of_days = [[] for i in range(8)]
    cut_of_day_timestamps = [[] for i in range(8)]
    start = 9
    new_day_length = int(19*60/minute_interval)
    prod_out = []
    timestamps_out = []
    for i in range(len(cut_of_days)):
        cut_of_days[i].extend(days_look_at[i][start:start+new_day_length])
        cut_of_day_timestamps[i].extend(timestamps[i][start:start+new_day_length])
        prod_out.extend(cut_of_days[i])
        timestamps_out.extend(cut_of_day_timestamps[i])
        print(timestamps[i][start], timestamps[i][start+new_day_length], len(cut_of_days[i]))
    print(len(prod_out),len(timestamps_out),int(8*new_day_length))
    frame= {'prod':prod_out, 'timestamps':timestamps_out}
    filename_out = "june_july_august_" + str(minute_interval)+ "_gp_production.csv"
    df_out = pd.DataFrame(frame)
    df_out.to_csv(filename_out)
    plt.figure(4,(20,10))
    plt.plot(prod_out)
    plt.figure(5,(20,10))
    plt.plot(df_out['prod'].diff().tolist(),'yo', markersize=1)
    
    
def gp_run(minute_interval = 10):
    filename = "june_july_august_" + str(minute_interval)+ "_gp_production.csv"
    df = pd.read_csv(filename)
    print(df.head())
    day_length = int(19*60/minute_interval)
    prod_list = df['prod'].tolist()
    print(np.cov(prod_list))
    #pd.plotting.autocorrelation_plot(prod_list[0:day_length])
    cov = 0.09986 # cov = 0.09986420544298752
    start = 0#2*day_length
    X = np.array(df.index.tolist()[start:7*day_length]).reshape(-1,1)
    y = np.array(df['prod'].tolist()[start:7*day_length]).reshape(-1,1)
    kper = cov**2 * ExpSineSquared(length_scale=0.4, periodicity=day_length)  # seasonal component
    klin = DotProduct(sigma_0 = 100)
    kernel = kper * klin
    
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1,
                              optimizer=None, normalize_y=True)
    gp.fit(X, y)

    print("Kernel: %s" % gp.kernel_)
    print("Log-marginal-likelihood: %.3f" % gp.log_marginal_likelihood(gp.kernel_.theta))
    X_ = np.array(df.index.tolist()[7*day_length:8*day_length]).reshape(-1,1)
    y_pred, y_std = gp.predict(X_, return_std=True)
    y_actual = np.array(prod_list[7*day_length:8*day_length]).reshape(-1,1)
    #y_persistence = np.array(prod_list[6*day_length:7*day_length]).reshape(-1,1)
    
    plt.figure(1,(20,10))
    plt.scatter(X, y, c='k',s=3, label = 'history')
    plt.plot(X_,y_actual, 'b', label = 'actual')
    #plt.plot(X_,y_persistence, 'y')
    plt.plot(X_, y_pred, 'r', label = 'forecast')
    ticks = np.arange(912,step=day_length/4)
    label_lib = ['20.30/01.30', '06.15', '11', '15.45']
    labels = []
    for i in range(8):
        labels.extend(label_lib)
    plt.xticks(ticks=ticks, labels=labels, rotation=75)
    plt.xlabel('Hour of day')
    plt.ylabel('Normalized Production')
    plt.title('A')
    plt.legend(loc='upper left')
    
