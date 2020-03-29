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
y = labels[:1000]  # y = labels

# visulaize distribution of training samples
# not necesarry, just a nice feature to get an overview
sns.countplot(y)

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


def cross_validation_linear(X,y,n=4):
    kf = KFold(n_splits=n)   # define how you want to split data (split into quarters)
    accuracy_s = []          # to store the accurycies for the different runs / sets
    
    
    # Do the cross validation 
    for train, test in kf.split(X):  # generate the 4 diffenrent train/test combinations
                                     # and for each of them do the svm and get the accuracy
        #print("%s %s" % (len(train), len(test)))
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        
        #print('SVM Classifier with gamma = 0.1; Kernel = linear')
        classifier = SVC(C = 1, gamma=0.1, kernel='linear', random_state = 0)
        classifier.fit(X_train,y_train)
        
        y_pred = classifier.predict(X_test)
        
        model_acc = classifier.score(X_test, y_test)
        #print('\nSVM Trained Classifier Accuracy: ', model_acc)
    
        test_acc = accuracy_score(y_test, y_pred)
        accuracy_s.append(test_acc)
        #print('\nPredicted Values: ',y_pred)
        #print('\nAccuracy of Classifier on Validation Images: ',test_acc)
    
        conf_mat = confusion_matrix(y_test,y_pred)
        #print('\nConfusion Matrix: \n',conf_mat)
    
    print(sum(accuracy_s)/4)
    
#cross_validation_lineark(X,y,4)

#------------------------------------------------------------------------------

# Paramter optimization - using gridsearch
    # the example on the scikit-learn webpage was foolowed for this part:
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html#sphx-glr-auto-examples-model-selection-plot-grid-search-digits-py


def tune_param_linear(param_dict, X,y):
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
            SVC(), tuned_parameters, scoring='%s_macro' % score
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
param_dictionary = [{'kernel': ['linear'], 'C': [0.1, 0.2, 0.23, 0.25, 0.27, 0.5, 1]}]

#tune_param_linear(param_dictionary)

# best parameter for linear kernel: C = 0.1  -> accuracy 0.87
    
# ----------------------------------------------------------------------------
    
# use SVM with linear kernel on TEST data

#X = train data
#y = labels
#test_test = the actual data you want to classify
# C_int = the C you want to use for your svm

def linear_svm_test(X,y,test_test, C_int=0.1):

    print('SVM Classifier with gamma = 0.1; Kernel = linear')
    classifier = SVC(C = C_int, kernel='linear', random_state = 0)
    classifier.fit(X,y)
    
    y_pred_t = classifier.predict(test_test)
    
    #model_acc = classifier.score(X_test, y_test)
    #print('\nSVM Trained Classifier Accuracy: ', model_acc)
    
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


#linear_svm_test(X,y,test_test, 0.1)  # accuracy of 0.92
    


# =============================================================================
# =============================================================================
# RBF Kernel - repeat steps from above with different kernel
# =============================================================================
# =============================================================================


# ----------------------------------------------------------------------------

# Cross validation
    
def cross_validation_rbf(X,y,n=4):

    kf = KFold(n_splits=4)   # define how you want to split data (split into quarters)
    accuracy_s = []          # to store the accurycies for the different runs / sets
    
    
    # Do the cross validation 
    for train, test in kf.split(X):  # generate the 4 diffenrent train/test combinations
                                     # and for each of them do the svm and get the accuracy
        #print("%s %s" % (len(train), len(test)))
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        
        
        # have to transform values ( else RBF kernel doesnt really work )
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)
        
        #print('SVM Classifier with gamma = 0.1; Kernel = linear')
        classifier = SVC(C=1,gamma = 'scale', kernel='rbf', random_state = 0)
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
    
    print(sum(accuracy_s)/4)  
    
    
#cross_validation_rbf(X,y,4)  # average accuarcy = 0.837

#------------------------------------------------------------------------------

# Paramter optimization - using gridsearch

tuned_parameters = [{'kernel': ['rbf'], 'C': [ 0.5, 1,2,3,4, 5,6,7,8, 10]},]

def tune_param_rbf(param_dict):
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
            SVC(), tuned_parameters, scoring='%s_macro' % score
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


a = [{'kernel': ['rbf'], 'C': [ 0.5, 1,2,3,4, 5,6,7,8, 10]}]

tune_param_rbf(a)

# best parameters for rbf kernle C = 8 accuracy 0.84


#------------------------------------------------------------------------------

def rbf_svm_test(X,y,test_test, C_int=0.1):

    print('SVM Classifier with gamma = 0.1; Kernel = linear')
    classifier = SVC(C = C_int, kernel='rbf', random_state = 0)
    X = sc_X.fit_transform(X)
    test_test = sc_X.transform(test_test)
    classifier.fit(X,y)
    
    y_pred_t = classifier.predict(test_test)
    
    #model_acc = classifier.score(X_test, y_test)
    #print('\nSVM Trained Classifier Accuracy: ', model_acc)
    
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


rbf_svm_test(X,y,test_test, 8)  # accuracy of 0.86






