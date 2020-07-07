# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 18:39:11 2020

@author: harshitm
"""


import numpy as np
import _pickle as cpickle
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

#function to read the data from file and sort them if required
def get_data(data_mat, sort_val):
    N, M = data_mat.shape
    if (sort_val):
        data_mat = np.array(sorted(data_mat, key = lambda a_entry : a_entry[-1]))
    x = np.zeros((N,M-1), dtype = int)
    y = np.zeros(N, dtype = int)
    x = data_mat[:,0:M-1]
    y = data_mat[:,M-1]
    for i in range(0, M-1):
        x[i] = x[i] / 255
    return x, y

#function to visualize the image
def plot_img_vec(img_vec, y):
    img_mat = np.zeros((28,28), dtype = int)
    k = 0
    for i in range(0,28):
        for j in range(0,28):
            img_mat[i][j] = img_vec[k]
            k+=1
    plt.imshow(img_mat)

#function used in pegasis SGD algorithm to find the sum of relevant set
def find_relevant_set(a_t, x_class_0, x_class_1, w):
    sum_ = np.zeros(w.shape)
    for i in a_t:
        if(-1.0*(np.dot(np.transpose(w),x_class_0[i,])) < 1.0):
            sum_ += (-1.0)*x_class_0[i,:]  
        if(1.0*(np.dot(np.transpose(w),x_class_1[i,:])) < 1.0):
            sum_ += (1.0)*x_class_1[i,:]
    return sum_

#Stochastic Gradient Descent using Pegasos Algo
def sgd_(x_class_0, x_class_1):

    #add extra 1 in input x to compute b in pegasos
    x_class_0 = np.insert(x_class_0, 0, 1, 1)
    x_class_1 = np.insert(x_class_1, 0, 1, 1)
    N0, M0 = x_class_0.shape
    N1, M1 = x_class_1.shape
    param_w = []
    batch = 100
    T = 100
    C = 1.0
    Lambda = 1/C*batch  
    w = np.zeros(M0)
    for t in range(1,T):
        a_t = np.random.randint(N0, size = batch)
        sum_ = find_relevant_set(a_t, x_class_0, x_class_1, w)
        n_t = 1/ (Lambda * t)
        w = (1 - 1/t) * w + (n_t/batch)*sum_
        param_w.append(w)
        
    param_w = np.array(param_w)
    return(param_w[-1])

#function to caclculate the parameters for all the classes using SGD
def find_all_class_param_(x_train, y_train):
    svm_train_file_name = "objs\svm_train_file"
    if os.path.exists(svm_train_file_name):
        fileObj = open(svm_train_file_name, 'rb')
        arr = cpickle.load(fileObj)
        params_comb_val = arr[0]
        params_w = arr[1]
        return params_comb_val, params_w
    
    N,M = x_train.shape
    params_comb_val = {}
    params_w = {}
    i = 0
    counter = 0
    while (i < N):
        j = i + 2000
        while( j < N):
            x_class_0 = x_train[i:i+2000,:]
            x_class_1 = x_train[j:j+2000,:]
            params_comb_val[counter] = [y_train[i], y_train[j]]
            params_w[counter] = sgd_(x_class_0, x_class_1)
            j += 2000
            counter = counter + 1
        i = i+ 2000
    fileObj = open(svm_train_file_name, 'wb')  
    arr = [params_comb_val, params_w]
    cpickle.dump(arr, fileObj)    
    return params_comb_val, params_w

#function used by predict function to predict the output of the test input 
def compute_pred(x_vec, params_comb_val, params_w):
    y_pred = {}
    max_y = 0
    for i in range(0,len(params_comb_val)):
        w_vec = params_w[i]
        b = w_vec[0]
        w_vec = w_vec[1:]
        y = params_comb_val[i]
        if(np.dot(np.transpose(w_vec), x_vec) + b < 0):
            y_pred[y[0]] = y_pred.get(y[0],0.0) + 1
        else:
            y_pred[y[1]] = y_pred.get(y[1],0.0) + 1
            
    max_y = max(y_pred.values())
    temp = [k for k, v in y_pred.items() if v == max_y]
    return(max(temp))
        
#function to predict the final output by computing the prediction for each class
def predict(x, params_comb_val, params_w):
    N, M = x.shape
    result = []
    for i in range(0,N):
        x_vec = x[i,:]
        pred = compute_pred(x_vec, params_comb_val, params_w)
        result.append(pred)
    return result

#function to find the accuray of the predicted results
def find_accuracy(y, result):
    N = len(y)
    sum = 0
    for i in range(0,N):
        if y[i] == result[i]:
            sum += 1
    return sum/N

#function to compute the confusion matrix
def create_conf_matrix(expected, predicted):
    conf_mat = np.zeros((10,10), dtype = int)
    N = len(expected)
    for i in range(0,N):
        conf_mat[int(expected[i])][int(predicted[i])] +=1
        
def main():
    start_time = time.time()
    train_data = pd.read_csv("train.csv",header=None)
    test_data = pd.read_csv("test.csv",header=None)
    x_train, y_train = get_data(train_data.values, True)
    x_test, y_test = get_data(test_data.values, False)
    params_comb_val , params_w = find_all_class_param_(x_train, y_train)
    x_train, y_train = get_data(train_data.values, False)
    result_test = predict(x_test, params_comb_val, params_w) 
    accuracy = find_accuracy(result_test, y_test)
    print(("{0:.4f}".format(accuracy)))
    print("--- %s seconds ---" % (time.time() - start_time))
    
if __name__ == "__main__":
    main()
