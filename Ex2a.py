#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:56:59 2020

@author: anikajohn
"""

#import matplotlib.pyplot as plt
import numpy as np 
#import pandas as pd
#import csv


image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
data_path = "mnist-csv-format/"
train = np.loadtxt(data_path + "mnist_train.csv", 
                        delimiter=",")
test = np.loadtxt(data_path + "mnist_test.csv", 
                       delimiter=",") 

#test_data[:10]

#print(test_data)

from sklearn import datasets, svm, metrics

short_train = train[0 :2000, :]
#print(len(short_train)) = 2000

st_2 = train[0:3000]

short_test = test[0:100, 1:]
test_lables = test[0:100, 0]



s_t_lables = short_train[:, 0] #only lables
s_t_pix = short_train[:, 1:] #only pixels 

st_l2 = st_2[:,0]
st_p2 = st_2[:,1:]

#print(s_t_pix.shape)

#The multiclass support is handled according to a one-vs-one scheme

#classifier1 = svm.SVC(C=200,kernel='rbf',gamma=6)

lin_clf = svm.LinearSVC(C= 1)


#classifier1.fit(s_t_pix,s_t_lables)

lin_clf.fit(st_p2,st_l2) #training step 


#x=classifier1.predict(short_test)

x = lin_clf.predict(short_test) #testing step 

print(x)



def linear_classifier (lables_train, pixel_train, pixel_test,c):
    
    lin_clf = svm.LinearSVC(C=c)
    lin_clf.fit(pixel_train,lables_train)
    predictions = lin_clf.predict(pixel_test)
    return predictions 

h =linear_classifier(st_l2, st_p2, short_test,0.2)
print(h)


#print(short_train[0:2000, 0])

def accuracy (lables, predictions):

    l = 0

    for i in range(len(predictions)):
        if predictions[i] == lables[i]: #compare if prediction is same as lable 
            l = l+1

    accuracy = l/len(predictions)

    return accuracy

print(accuracy(test_lables, h))


def cross_val_lin(train, C): 
    
    #devide the data set in 4 parts; each is once used as test set 
    
    test_1 = train[0:15000,1:] #only the pixesl
    lables_1 = train[0:15000,0] #only the lables 
    
    
    train_1 = train[15000:, 1:]
    train_l1 = train[15000:, 0]
      
    h1 =linear_classifier(train_l1, train_1, test_1,C)
    a1 = accuracy(lables_1, h1)
        
    #---------------------------
    
    test_2 = train[15000:30000,1:] #all test pixels; assume 15000 incl. 30000 not 
    lables_2 = train[15000:30000,0]
     
    train_2 = train[:15000,:] #15000 not incl.
    
    train_2b =train[30000:,:] #still pixels & lables
    
    train_2= np.concatenate((train_2, train_2b))
    
    train_l2 = train_2[:,0] #only lables
    train_2 = train_2[:, 1:] #only pixels
    
    h2 =linear_classifier(train_l2, train_2, test_2,C)
    a2 = accuracy(lables_2, h2)
    
    #---------------------------
    
    test_3 = train[30000:45000,1:]
    lables_3 = train[30000:45000,0]
    
    train_3 = train[:30000,:]
    
    train_3b = train[45000:,:]
   
    train_3= np.concatenate((train_3, train_3b))
    
    train_l3 = train_3[:,0]
    train_3 = train_3[:,1:]
    
    
    h3 =linear_classifier(train_l3, train_3, test_3,C)
    a3 = accuracy(lables_3, h3)
  
    
    #---------------------------
    
    test_4 = train[45000:,1:]
    lables_4 = train[45000:,0]
  
    
    train_4 = train[:45000, 1:]
    train_l4 = train[:45000, 0]

    
    h4 =linear_classifier(train_l4, train_4, test_4,C)
    a4 = accuracy(lables_4, h4)

    
    #now average accuracy
    
    av_acu = (a1+a2+a3+a4)/4
    
    return av_acu
    


b = cross_val_lin(train,1)
print(b) #0.8491

c = cross_val_lin(train,2)
print(c) #0.8449 worse than C=1 so go below C=1

d = cross_val_lin(train,0.5)####
print(d) #0.8522

e = cross_val_lin(train,0.25)#####
print(e) #0.8718

f = cross_val_lin(train,0.1)
print(f) #0.85505; got worse than before

g = cross_val_lin(train,0.2)
print(g) #0.8433

h = cross_val_lin(train,0.225)
print(h) #0.8543


i = cross_val_lin(train,0.24)
print(i) #0.8540

j = cross_val_lin(train,0.3)
print(j) #0.8477

k = cross_val_lin(train,0.27)
print(k)# 0.8627

l = cross_val_lin(train,0.26)
print(l) #0.8576


#not really an elegant solution to do this all by hand, maybe worth implementing
#an actual optimization algorithm
#0.25 seems to be the best possible value for C, also linear model over all
#far from perfect 












#C = 1
#
#
#test_1 = train[0:15000,1:] #only the pixesl
#lables_1 = train[0:15000,0] #only the lables 
#
#print(test_1.shape)
#print(lables_1.shape)
#
#train_1 = train[15000:, 1:]
#train_l1 = train[15000:, 0]
#
#print(train_1.shape)
#print(train_l1.shape)
#
#
#
#h1 =linear_classifier(train_l1, train_1, test_1,C)
#a1 = accuracy(lables_1, h1)
#
#print(a1)
#
##---------------------------
#
#test_2 = train[15000:30000,1:] #all test pixels; assume 15000 incl. 30000 not 
#lables_2 = train[15000:30000,0]
#
#print(test_2.shape)
#print(lables_2.shape)
#
#train_2 = train[:15000,:] #15000 not incl.
#print(train_2.shape)
#
#train_2b =train[30000:,:] #still pixels & lables
#print(train_2b.shape)
#
#train_2= np.concatenate((train_2, train_2b))
#print(train_2.shape)
#
#train_l2 = train_2[:,0] #only lables
#print(train_l2.shape)
#train_2 = train_2[:, 1:] #only pixels
#print(train_2.shape)
#
#h2 =linear_classifier(train_l2, train_2, test_2,C)
#a2 = accuracy(lables_2, h2)
#print(a2)
#
##---------------------------
#
#test_3 = train[30000:45000,1:]
#lables_3 = train[30000:45000,0]
#
#print(test_3.shape)
#print(lables_3.shape)
#
#
#train_3 = train[:30000,:]
#print(train_3.shape)
#
#train_3b = train[45000:,:]
#print(train_3b.shape)
#
#train_3= np.concatenate((train_3, train_3b))
#print(train_3.shape)
#
#train_l3 = train_3[:,0]
#train_3 = train_3[:,1:]
#print(train_l3.shape)
#print(train_3.shape)
#
#
#h3 =linear_classifier(train_l3, train_3, test_3,C)
#a3 = accuracy(lables_3, h3)
#print(a3)
#
##---------------------------
#
#test_4 = train[45000:,1:]
#lables_4 = train[45000:,0]
#
#print(test_4.shape)
#print(lables_4.shape)
#
#train_4 = train[:45000, 1:]
#train_l4 = train[:45000, 0]
#
#print(train_4.shape)
#print(train_l4.shape)
#
#h4 =linear_classifier(train_l4, train_4, test_4,C)
#a4 = accuracy(lables_4, h4)
#
#print(a4)
#
##now average accuracy
#
#av_acu = (a1+a2+a3+a4)/4
#
#print(av_acu)








