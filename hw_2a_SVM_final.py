# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:41:40 2020

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 16:53:45 2020

@author: Admin
"""

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix

import sklearn
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore',category=DeprecationWarning)
warnings.filterwarnings(action='ignore',category=FutureWarning)

#print(sklearn.__version__)  # 0.22.1



# https://www.kaggle.com/soumya044/mnist-digit-recognizer-using-kernel-svm


# =============================================================================
# Load data
# =============================================================================


with open('train.csv', 'r') as f:
    reader = csv.reader (f)
    data = list(reader)
    matrix = np.array (data, dtype = int)  # with labels
    samples = matrix[:,1:]  # no labeles
    labels = matrix[:,0]
    
#print(samples.shape)

X = samples[:1000]  # X = features
print(X)
y = labels[:1000]  # y = labels

# visulaize distribution of training samples
# not necesarry, just a nice feature to get an overview

sns.countplot(labels)


with open('test.csv', 'r') as t:
    reader = csv.reader(t)
    data = list(reader)
    matrix_t = np.array (data, dtype = int)  # with labels
    samples_t = matrix_t[:,1:]  # no labeles
    labels_t = matrix_t[:,0]
    

test_test = samples_t[:100]
test_truth = labels_t[:100]

#print(test_test.shape)

# =============================================================================
# linear kernel
# =============================================================================
from sklearn.model_selection import KFold


def cross_validation_linear(X,y,C,n=4):
    kf = KFold(n_splits=n)   # define how you want to split data (split into quarters)
    accuracy_s = []          # to store the accurycies for the different runs / sets
    
    classifier = SVC(C, gamma=0.1, kernel='linear', random_state = 0)
    # Do the cross validation 
    for train, test in kf.split(X):  # generate the 4 diffenrent train/test combinations
                                     # and for each of them do the svm and get the accuracy
        #print("%s %s" % (len(train), len(test)))
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        
        #print('SVM Classifier with gamma = 0.1; Kernel = linear')
        # classifier = SVC(C = 1, gamma=0.1, kernel='linear', random_state = 0)
        classifier.fit(X_train,y_train)
        
        y_pred = classifier.predict(X_test)
        
        #model_acc = classifier.score(X_test, y_test)
        #print('\nSVM Trained Classifier Accuracy: ', model_acc)
    
        test_acc = accuracy_score(y_test, y_pred)
        accuracy_s.append(test_acc)
        #print('\nPredicted Values: ',y_pred)
        #print('\nAccuracy of Classifier on Validation Images: ',test_acc)
    
        #conf_mat = confusion_matrix(y_test,y_pred)
        #print('\nConfusion Matrix: \n',conf_mat)
    
    print('Accuracies of the different corssvaidations', accuracy_s)
    print(accuracy_s)
    print('average accuracy', sum(accuracy_s)/4)
    return classifier
    
#c = cross_validation_linear(X,y,4)

# Accuracies of the different corssvaidations [0.84, 0.9, 0.896, 0.872]
# average accuracy 0.877

#------------------------------------------------------------------------------

# Paramter optimization - using gridsearch
    # the example on the scikit-learn webpage was foolowed for this part:
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html#sphx-glr-auto-examples-model-selection-plot-grid-search-digits-py


def tune_param_linear(param_dict, X,y, classifier):
    # (list) dictionary(s) with values of parameters you want to test
    tuned_parameters = param_dict
    #tuned_parameters = [{'kernel': ['linear'], 'C': [0.1, 0.2, 0.23, 0.25, 0.27, 0.5, 1]},]
    
    # returns best parameters (of the tested) and also which 
    # of the tested kernels is better (if you test more than one kernel at once,
    # here I only thest the linear kernel)
    # along with some other informative stuff
    
    scores = ['precision', 'recall']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
    
        clf = GridSearchCV(
            classifier, tuned_parameters, scoring='%s_macro' % score
        )
        clf.fit(X_train, y_train)
    
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()
    
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

# use this format as input for the param_dict  (used dict below)
param_dictionary = [{'kernel': ['linear'], 'C': [0.1, 0.2, 0.25, 0.5, 1,100]}]

#tuned = tune_param_linear(param_dictionary, X,y,c)

# best parameter for linear kernel: C = 0.1  -> accuracy 0.87
    
# ----------------------------------------------------------------------------
    
# use SVM with linear kernel on TEST data

#X = train data
#y = labels
#test_test = the actual data you want to classify
# C_int = the C you want to use for your svm

def linear_svm_test(X,y,C,test_test,test_truth):

    print('SVM Classifier with gamma = 0.1; Kernel = linear')
    #classifier = SVC(C = C_int, kernel='linear', random_state = 0)
    classifier = cross_validation_linear(X, y, C,4)
    classifier.fit(X,y)
    
    y_pred_t = classifier.predict(test_test)
    
    #model_acc = classifier.score(X_test, y_test)
    #print('\nSVM Trained Classifier Accuracy: ', model_acc)
    accuracy_s = []
    test_acc = accuracy_score(test_truth, y_pred_t)
    accuracy_s.append(test_acc)
    print('\nPredicted Values: ',y_pred_t)
    print('\nAccuracy of Classifier on Validation Images: ',test_acc)
    
    conf_mat = confusion_matrix(test_truth,y_pred_t)
    print('\nConfusion Matrix: \n',conf_mat)
    
    
    # nice visual representation of the confusion matrix
    plt.matshow(conf_mat)
    plt.title('Confusion Matrix for Validation Data')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


#linear_svm_test(X,y,1,test_test,test_truth)  # accuracy of 0.92

# a = linear_svm_test(samples, labels, samples_t, labels_t, 0.1)
import time
# start = time.time()
#b = linear_svm_test(samples, labels,1, samples_t, labels_t)
# end = time.time()
# print(end-start)

# accuracy   0.9078728084794347
# Confusion Matrix: 
#  [[1463    0    7    5    2   12   13    0    5    0]
#  [   0 1651    3    5    1    3    2    3   10    1]
#  [  13   17 1331   29    8    6   16   12   20    3]
#  [  11   15   43 1360    2   59    1   10   32   11]
#  [   6   10   20    1 1324    0    4    4    5   36]
#  [  20    6   17   64   15 1174   18    2   30   13]
#  [  14    3   17    0   14   18 1385    0    6    0]
#  [   5    4   26    8   17    1    0 1449    3   77]
#  [  10   46   43  106    6   47    6    7 1190   13]
#  [   9    5    9   16   98    9    0   75   13 1292]]

# Accuracy of Classifier on Validation Images:  0.9080727951469902

# Confusion Matrix: 
#   [[1463    0    7    5    2   12   13    0    5    0]
#   [   0 1651    3    5    1    3    2    3   10    1]
#   [  13   17 1331   29    8    6   16   12   20    3]
#   [  11   15   43 1360    2   59    1   10   32   11]
#   [   6   10   20    1 1324    0    4    4    5   36]
#   [  20    6   17   65   15 1173   18    2   30   13]
#   [  14    3   17    0   14   18 1385    0    6    0]
#   [   5    4   26    8   17    1    0 1449    3   77]
#   [  10   46   43  100    7   47    6    8 1194   13]
#   [   9    5    9   16   98    9    0   75   13 1292]]
# 1117.4579079151154

# =============================================================================
# =============================================================================
# RBF Kernel - repeat steps from above with different kernel
# =============================================================================
# =============================================================================


# ----------------------------------------------------------------------------

# Cross validation
    
def cross_validation_rbf(X,y,C,n=4):

    kf = KFold(n_splits=4)   # define how you want to split data (split into quarters)
    accuracy_s = []          # to store the accurycies for the different runs / sets
    #print('SVM Classifier with gamma = 0.1; Kernel = linear')
    classifier = SVC(C,gamma = 'scale', kernel='rbf', random_state = 0)
    # Do the cross validation 
    for train, test in kf.split(X):  # generate the 4 diffenrent train/test combinations
                                     # and for each of them do the svm and get the accuracy
        #print("%s %s" % (len(train), len(test)))
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        
        
        # have to transform values ( else RBF kernel doesnt really work )
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)
        
        
        
        classifier.fit(X_train,y_train)
        
        y_pred = classifier.predict(X_test)
        
        model_acc = classifier.score(X_test, y_test)
        #print('\nSVM Trained Classifier Accuracy: ', model_acc)
    
        test_acc = accuracy_score(y_test, y_pred)
        accuracy_s.append(model_acc)
        #print('\nPredicted Values: ',y_pred)
        #print('\nAccuracy of Classifier on Validation Images: ',test_acc)
    
        conf_mat = confusion_matrix(y_test,y_pred)
        #print('\nConfusion Matrix: \n',conf_mat)
    print('Accuracy' , accuracy_s)
    print('Average accuracy', sum(accuracy_s)/4)  
    return classifier
    
    
#c_rbf =  cross_validation_rbf(X,y,4)  

#c3_rbf =  cross_validation_rbf(samples_t,labels_t,5,4) 

# before parameter tuning C = 1
# Accuracy [0.792, 0.872, 0.844, 0.84]
# Average accuracy 0.837


# after parameter tuning C = 5
# Accuracy [0.804, 0.872, 0.86, 0.844]
# Average accuracy 0.845

#------------------------------------------------------------------------------

# Paramter optimization - using gridsearch

tuned_parameters = [{'kernel': ['rbf'], 'C': [ 0.5, 1,2,3,4, 5,6,7,8, 10]},]

def tune_param_rbf(param_dict, classifier):
    # (list) dictionary(s) with values of parameters you want to test
    tuned_parameters = param_dict
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    
    scores = ['precision', 'recall']
    
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
    
        clf = GridSearchCV(
            classifier, tuned_parameters, scoring='%s_macro' % score
        )
        clf.fit(X_train, y_train)
    
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()
    
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()


a = [{'kernel': ['rbf'], 'C': [ 0.5, 1, 5,6,7,8, 10]}]

#tune_param_rbf(a, c_rbf)

# Best parameters set found on development set:

# {'C': 5, 'kernel': 'rbf'}

# 0.836 (+/-0.057) for {'C': 0.5, 'kernel': 'rbf'}
# 0.857 (+/-0.060) for {'C': 1, 'kernel': 'rbf'}
# 0.868 (+/-0.036) for {'C': 5, 'kernel': 'rbf'}
# 0.867 (+/-0.037) for {'C': 6, 'kernel': 'rbf'}
# 0.867 (+/-0.037) for {'C': 7, 'kernel': 'rbf'}
# 0.867 (+/-0.037) for {'C': 8, 'kernel': 'rbf'}
# 0.867 (+/-0.037) for {'C': 10, 'kernel': 'rbf'}

# best parameters for rbf kernle C = 5 


#------------------------------------------------------------------------------

def rbf_svm_test(X,y,C,test_test,test_truth):

    print('SVM Classifier  Kernel = rbf')
    #classifier = SVC(C = C_int, kernel='rbf', random_state = 0)
    classifier = cross_validation_rbf(X, y, C, 4)
    
    sc_X = StandardScaler()
    X = sc_X.fit_transform(X)
    test_test = sc_X.transform(test_test)
    classifier.fit(X,y)
    
    y_pred_t = classifier.predict(test_test)
    
    #model_acc = classifier.score(X_test, y_test)
    #print('\nSVM Trained Classifier Accuracy: ', model_acc)
    accuracy_s = []
    test_acc = accuracy_score(test_truth, y_pred_t)
    accuracy_s.append(test_acc)
    print('\nPredicted Values: ',y_pred_t)
    print('\nAccuracy of Classifier on Validation Images: ',test_acc)
    
    conf_mat = confusion_matrix(test_truth,y_pred_t)
    print('\nConfusion Matrix: \n',conf_mat)
    
    
    # nice visual representation of the confusion matrix
    plt.matshow(conf_mat)
    plt.title('Confusion Matrix for Validation Data')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    


#rbf_svm_test(X,y,1,test_test, test_truth)  # accuracy of 0.87#

# import time
# start = time.time()
# rbf_test = rbf_svm_test(samples, labels,1, samples_t, labels_t)
# end = time.time()
# print(end-start)

# SVM Classifier with gamma = 0.1; Kernel = rbf
# Accuracy [0.9512592592592592, 0.9481481481481482, 0.9545185185185185, 0.9473996147577419]
# Average accuracy 0.950331385170917

# Predicted Values:  [9 2 5 ... 7 6 9]

# Accuracy of Classifier on Validation Images:  0.9579361375908273

# Confusion Matrix: 
#   [[1476    0    9    1    1    2   11    1    4    2]
#   [   0 1655   10    1    3    1    1    2    5    1]
#   [   5    4 1408    8    5    0    9    6    8    2]
#   [   1    5   23 1472    1   12    3   13    8    6]
#   [   3    6   18    0 1349    2    4    0    2   26]
#   [   6    3   13   22    5 1284   11    2    8    5]
#   [  11    3   17    0    2    8 1412    0    4    0]
#   [   2    2   30    1    6    1    0 1530    2   16]
#   [   4   16   19    9    6   21    3    3 1388    5]
#   [   6    5   22   20   21    5    0   40   11 1396]]
# 2972.11518907547

# Accuracy [0.9512592592592592, 0.9481481481481482, 0.9545185185185185, 0.9473996147577419]
# Average accuracy 0.950331385170917

# Predicted Values:  [9 2 5 ... 7 6 9]

# Accuracy of Classifier on Validation Images:  0.9579361375908273

# Confusion Matrix: 
#  [[1476    0    9    1    1    2   11    1    4    2]
#  [   0 1655   10    1    3    1    1    2    5    1]
#  [   5    4 1408    8    5    0    9    6    8    2]
#  [   1    5   23 1472    1   12    3   13    8    6]
#  [   3    6   18    0 1349    2    4    0    2   26]
#  [   6    3   13   22    5 1284   11    2    8    5]
#  [  11    3   17    0    2    8 1412    0    4    0]
#  [   2    2   30    1    6    1    0 1530    2   16]
#  [   4   16   19    9    6   21    3    3 1388    5]
#  [   6    5   22   20   21    5    0   40   11 1396]]
# 1311.4284188747406


