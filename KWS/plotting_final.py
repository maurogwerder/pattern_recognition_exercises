#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 08:46:04 2020
@author: anikajohn
"""
#f = open("with_Result.txt")
#print(f.read())
#
#l = f.read()
#print(l)

import re 

filename = "with_Result.txt"

with open(filename) as f:
    lines = f.read()

result = eval(lines)

#print(type(result))

only_words = [b for a, b in result]
#print(only_words)
#
#short_list = only_words[:5]
#print(short_list)
#
#rest_list = only_words[5:]
#print(rest_list)
#
#
#tp = re.compile("w-i-t-h_*.")
#TP_list = list(filter(tp.match, short_list)) 
#
#TP = len(TP_list)
#FP = len(short_list)-TP
#
#
#fn = re.compile("w-i-t-h_*.")
#FN_list = list(filter(fn.match, rest_list)) 
#
#print(FN_list)
#
#FN = len(FN_list)
#
#recall = TP/(TP+FN)
#precision = TP/(TP+FP)
#
#print(recall, precision)

#------LOOP---------


precision= list()
recall= list()

for i in range(1,1293): #go in 1er steps 
    
    short_list = only_words[:i]
    #print(short_list)
    
    rest_list = only_words[i:]
   # print(rest_list)
    
    
   #ADD WORD PATTERN HERE!!!!!!!!!!!
   
    tp = re.compile("w-i-t-h_*.")
    TP_list = list(filter(tp.match, short_list)) 
    
    TP = len(TP_list)
    FP = len(short_list)-TP
    
    
    fn = re.compile("w-i-t-h_*.")
    FN_list = list(filter(fn.match, rest_list)) 
    
    #print(FN_list)
    
    FN = len(FN_list)
    
    rec = TP/(TP+FN)
    prec= TP/(TP+FP)
    
    precision.append(prec)
    recall.append(rec)


#print(len(recall))
#print(len(precision))

import matplotlib.pyplot as plt

plt.plot(recall,precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
