#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:10:43 2020

@author: anikajohn
"""
import numpy as np 
import matplotlib.pyplot as plt
from scipy.misc import imread

#black = 0
#white = 1

#test vector 
vector = np.array([1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1])
#print(vector)

#-----Black white transitions-------------------

def bw_transitions(vector):
    transitions= 0 
    
    #print(len(vector))
    
    for i in range(len(vector)-1):
        #print(vector[i])
        if vector[i] != vector[i+1]:
            transitions = transitions +1
    
    bw_transitions = transitions/2
    
    return bw_transitions 

bla = bw_transitions(vector)
print(bla)
  
print(bw_transitions)

#----fraction of black pixels---------------------

def fract_black(vector):
    
    black_ind = np.where(vector==0)
    black = len(black_ind[0])
    fr_black = black/len(vector)
    
    return fr_black

    
    
    
#print(len(vector))
#print(fract_black(vector))


# =============================================================================
# Making the strings 
# =============================================================================

word_2 = plt.imread('word_2.png')
plt.imshow(word_2)
print(word_2.shape)

rows = int(word_2.shape[0])
cols = int(word_2.shape[1])

def binarize(img,row,col):
    
    for i in range(row):
        for j in range(col):
            
            if img[i][j] < 0.5:
                img[i][j] = 0
            
            else:
                img[i][j] = 1 
    
    return img 
        
word_2 =  binarize(word_2,rows,cols)  
plt.imshow(word_2)   
        
    

def window_trimmer(window):
    i = 0

    while i < len(window):
        if window[i] != 0:
            window = window[i:]
            i = len(window)
        i += 1

    j = len(window) - 1

    while j > 0:
        if window[j] != 0:
            window = window[0: j + 1]
            j = 0
        j = j - 1
    
    return(window)
    
#t = word_2[:,90]
#t_vec = window_trimmer(t)
#t_trans = bw_transitions(t_vec)
#print((t_vec))
#print(t_trans)
##plt.imshow(t_vec)
#
n_windows = word_2.shape[1]
#print(n_windows)

trans = list() #vector with number of black white transitions per window
black_frac = list()#vector with fraction of black pixels pwr window 


for i in range(n_windows):
    
    window = word_2[:,i]
    vector_w = window_trimmer(window)
    
    bw_tran = bw_transitions(vector_w)
    trans.append(bw_tran)
    
    black = fract_black(vector_w)
    black_frac.append(black)
    
    #print(type(window))

print(len(black_frac))


