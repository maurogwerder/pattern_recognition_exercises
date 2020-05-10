# -*- coding: utf-8 -*-
"""
Created on Sat May  9 18:31:32 2020

@author: danie
"""


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


import numpy as np

import matplotlib.pyplot as plt


import os
import re 



all_key_words = list()
with open("keywords.txt", "r") as key:
    for i in key:
        all_key_words.append(i.strip())
#reduces to only one word

for keys in all_key_words:

    #keys = all_key_words[0]
    
    
    
    print(keys)
    
    missed_words = 0
    word_nr = 0
    
    
    
        
    word_nr +=1
    
    word = keys
    
    directories = os.listdir()
    pat = re.compile(word+"_.*\.txt")
    features = list()
    for i in directories:
        if pat.findall(i) != []:
            features.append(pat.findall(i))
    
    if features != []:
        print("word_nr", word_nr)
        precision_precision = list()
        recall_recall= list()
    
        for file in features:
        
            filename = file[0]
            
            with open(filename) as f:
                lines = f.read()
            
            result = eval(lines)
            result = result
            
        
            
            for j in result:
            
            
                only_words = [b for a, b, c in j]
        
                
                #------LOOP---------
                precision= list()
                recall= list()    
        
                
        
                
                for i in range(1,1293): #go in 1er steps 
                    
                    short_list = only_words[:i]
                    #print(short_list)
                    
                    rest_list = only_words[i:]
                   # print(rest_list)
                    
                    
                   #ADD WORD PATTERN HERE!!!!!!!!!!!
                   
                    tp = re.compile(word)
                    TP_list = list(filter(tp.match, short_list)) 
                    
                    TP = len(TP_list)
                    FP = len(short_list)-TP
                
                    
                    fn = re.compile(word)
                    FN_list = list(filter(fn.match, rest_list)) 
                    
                    #print(FN_list)
                    
                    FN = len(FN_list)
                    #print(TP, FN, FP)
                    if (TP+FN) == 0:
                        rec = 0
                    else:
                        rec = TP/(TP+FN)
                     
                    if (TP+FP) == 0:    
                        prec = 0
                    else:
                        prec= TP/(TP+FP)
                    
                    
                    
                    precision.append(prec)
                    recall.append(rec)
           
                precision_precision.append(precision)
                recall_recall.append(recall)    
    
    
    
        precision_precision = np.asarray(precision_precision)
        recall_recall = np.asarray(recall_recall)
        
        precision_mean = np.mean(precision_precision, axis=0)
        recall_mean = np.mean(recall_recall, axis=0)
        
         
        
        #print(len(recall))
        #print(len(precision))
        
        
        
        
        #plotting the result
        plt.plot(recall_mean,precision_mean)
        plt.title(keys)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig("pictures/" + keys)
        plt.show()
        plt.clf()
        print("missed_words: ", missed_words)
    
    
    
    
    
    
    
    
    else:
        print("word_nr", word_nr, " missed! ")
        missed_words += 1
        
