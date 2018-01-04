# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 16:31:51 2017

@author: Lily
"""

import math
import pandas as pd
import numpy as np
import random
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.neighbors import DistanceMetric


#using matplotlib’s ggplot style
plt.style.use('ggplot')
import seaborn as sns
sns.set(color_codes=True)


def gaussian(dist, sigma):          #Gaussian kernel
    #tempg= np.array(np.power(-dist,2)/(2*sigma**2),dtype=np.float32)
    dist = np.array(dist, dtype = np.float32) 
    tempg= np.power((2*np.pi), -0.5)*np.exp(-np.power(dist, 2)/2*sigma**2)
    #nv = np.array(1/np.abs(dist))  #inverse kernel
    #weight = np.array([0.25, 0.25, 0.25, 0.25]) #average kernel
    return tempg

def loadDataset(dataArr, split, trainSet = [], testSet = [], testId = []):
    '''
    :dataArr is a dataFrame
    :split is a float
    '''
    index_i = dataArr.index.get_values()

    for x in index_i:
        if random.random() < split:
            trainSet.append(dataArr.loc[x, :])
        else:
            testSet.append(dataArr.loc[x, :])   
            testId.append(x)
#    positive_train = trainSet[trainSet['close_price'] >= 0]
#    positive_test = testSet[testSet['close_price'] >= 0]
#    trainSet = positive_train     #remove negative closing price data
#    testSet = positive_test
    
    return (trainSet, testSet, testId)
    

def knn(k, data, index, dataArray):       #knn solver
    result = []                #store 4nn predicted housing price

    for i in index:            
        temp = data[:, 2] < dataArray[i, 2]
        priori = data[temp]
        x1 = dataArray[i, 0]
        #print(x1)
        y1 = dataArray[i, 1]
        #print(y1)
        
        #calculate distance
        temp1 = np.array(0.5*np.absolute(priori[:, 0]-x1),dtype=np.float32)
        temp2 = np.array(0.5*np.absolute(priori[:, 1]-y1),dtype=np.float32)
        temp3 = np.array(priori[:, 0], dtype = np.float32)
        dista = 2*np.arcsin(np.sqrt(np.power(np.sin(temp1), 2)*np.abs(np.cos(x1)*np.cos(temp3))*np.power(np.sin(temp2), 2)))
        
#        radius = 3959 #earth radius
#        dlat = np.array(np.absolute(priori[:, 0]-x1),dtype=np.float32)
#        dlon = np.array(np.absolute(priori[:, 1]-y1),dtype=np.float32)
#        
#        a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(x1)) * np.cos(np.radians(np.array(priori[:,0], dtype = np.float32))) * np.sin(dlon/2) * np.sin(dlon/2)
#
#        c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
#        dista = radius * c
#        lat1 = np.array(priori[:,0], dtype = np.float32)
#        lon1 = np.array(priori[:,1], dtype = np.float32)
#        dista = np.sqrt(np.sum([(lat1 - x1)**2,(lon1 - y1)**2]))
        

        new = np.insert(priori, 5, dista, axis=1)   #add distance to the right of the array as a row
        new = new[new[:,5].argsort()]        #sort the array by the row
        nn = new[0:k, [3, 5]]           #k=4m keep the column of housing price and distance
        #print(nn.shape)
        result.append(np.dot(gaussian(nn[:,1], 10.0), nn[:,0]))      #calculate weighted results
        
                      
#        
#        result.append(np.dot(weight.T, nn[:,0]))
        
    return result


##measure performance
##RMAE median of absolute value of (prediction - actual value) over actual value
def performanceEval(testSet, testSetPred):
    '''
    this part is for metrics of performance
    
    '''
    
    tempEval = np.array(np.abs(testSet - testSetPred)/testSet, dtype = np.float32)
    RMAE = np.median(tempEval)
    #print("RMSE for kNN predictor is :", RMSE)
    return RMAE
    
#    index = np.arange(1, len(testSet)+1) 
#    plt.figure(figsize=(20,10))
#    plt.plot(index, testSet, color='b', linestyle='-', marker='*', label='Actual Price')
#    plt.plot(index, testSetPred, color='r', linestyle='--', marker='d', label='Prediction')
#    plt.legend(loc='best', fancybox=True, shadow=False, prop={'size':9})
#    plt.xlabel('Time', fontsize=9)
#    plt.ylabel('RMSE', fontsize=9)
#    plt.title('Temporal Prediction Performance')
#    plt.xlim(1, len(testSet)+1)
#    plt.grid()
#    
#                                  

#print(knn(4, part, index))

def main():
    #data = pd.read_csv("/Users/Alex/Downloads//data.csv")  #load data
    data = pd.read_csv("I://data.csv")  # read data
    data['close_date'] = pd.to_datetime(data['close_date'])   #convert to stardard datetime         
    data['lan_lon'] = [list(a) for a in zip(data['latitude'], data['longitude'])]
    
    
    positive = data[data['close_price'] >= 0]
    #positive = positive[positive['longitude'] < 0]
    df = positive   #remove negative closing price data
    dataArray = np.array(df)
    split = 0.9
    dpart = df.iloc[0:1000, :]
    
    error = []
    for i in range(10):
    
        tr_te = loadDataset(dpart, split, [], [], [])
        
        te_index = tr_te[2]
        train = np.array(tr_te[0])
        test = np.array(tr_te[1])
        #print(te_index)
        k = 2   #k value
         
        testSetPred = knn(k, train, te_index, dataArray)  #knn solver
        error.append(performanceEval(test[:,3], np.array(testSetPred)))   #performance metrics
        #print(test[0:2, 3])
        #print(np.array(testSetPred)[0:2])
      
    print("MRAE: %0.3f (+/- %0.3f)" % (np.mean(error), np.std(error)))

main()


  
  
  
  
  
  
  