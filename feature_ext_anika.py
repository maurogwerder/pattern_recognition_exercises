#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:10:43 2020

@author: anikajohn
"""
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
#from scipy.misc import imread
from os.path import isfile, join
from os import listdir


#black = 0
#white = 1

#test vector 
#vector = np.array([1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1])
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


def single_lc(col):
    for row in reversed(range(0, len(col))):
        if col[row] == 0:
            fraction = row / len(col)
            # print(fraction)
            return 1 - fraction, row
        elif row == 0:
            return None, None


def single_uc(col):
    for row in range(0, len(col)):
        if col[row] == 0:
            fraction = row / len(col)

            return 1 - fraction, row
            break
        elif row == (len(col) - 1):
            return None, None


def inbound_ratio(col, uc, lc):
    col = col[uc:lc]
    if len(col) == 0:
        return 0
    ratio = sum(col == 0) / len(col)
    return ratio


#bla = bw_transitions(vector)
#print(bla)
  
#print(bw_transitions)

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
"""
word_2 = plt.imread('word_2.png')
word_40 = plt.imread('word_40.png')
word_46 = plt.imread('word_46.png')


plt.imshow(word_2)
print(word_2.shape)

#rows

rows = int(word_2.shape[0])
cols = int(word_2.shape[1])
"""


def binarize(img):
    
    row = int(img.shape[0])
    col = int(img.shape[1])
    
    for i in range(row):
        for j in range(col):
            
            if img[i][j] < 0.5:
                img[i][j] = 0
            
            else:
                img[i][j] = 1 
    
    return img 
        
#word_2 =  binarize(word_2)
#plt.imshow(word_2)
#word_40 = binarize(word_40)
#word_46 = binarize(word_46)
#plt.imshow(word_46)

    

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
#n_windows40 = word_40.shape[1]
#n_windows46 = word_46.shape[1]
#n_windows2 = word_2.shape[1]

#trans40 = list() #vector with number of black white transitions per window
#black_frac40 = list()#vector with fraction of black pixels pwr window

"""
for i in range(n_windows40):
    
    window40 = word_40[:,i]
    vector_w40 = window_trimmer(window40)
    
    bw_tran40 = bw_transitions(vector_w40)
    trans40.append(bw_tran40)
    
    black40 = fract_black(vector_w40)
    black_frac40.append(black40)
    
    #print(type(window))


trans46 = list() #vector with number of black white transitions per window
black_frac46 = list()#vector with fraction of black pixels pwr window 


for i in range(n_windows46):
    
    window46 = word_46[:,i]
    vector_w46 = window_trimmer(window46)
    
    bw_tran46 = bw_transitions(vector_w46)
    trans46.append(bw_tran46)
    
    black46 = fract_black(vector_w46)
    black_frac46.append(black46)
    
    #print(type(window))


trans2 = list() #vector with number of black white transitions per window
black_frac2 = list()#vector with fraction of black pixels pwr window 


for i in range(n_windows2):
    
    window2 = word_2[:,i]
    vector_w2 = window_trimmer(window2)
    
    bw_tran2 = bw_transitions(vector_w2)
    trans2.append(bw_tran2)
    
    black2 = fract_black(vector_w2)
    black_frac2.append(black2)
    
    #print(type(window))
"""
def feature_extract(word):
    cols = int(word.shape[1])
    #word = binarize(word)
    trans = list()  # vector with number of black white transitions per window
    black_frac = list()  # vector with fraction of black pixels per window
    lc_frac_l = list()  # vector with the relative position of the lower contour per window
    uc_frac_l = list()  # vector with the relative position of the upper contour per window
    bound_frac = list()  # vector with fraction of black pixels within the contours per window
    for i in range(cols):
        window = word[:, i]
        vector = window_trimmer(window)

        bw_tran = bw_transitions(vector)
        trans.append(bw_tran)

        black = fract_black(vector)
        black_frac.append(black)

        lc_frac, lc = single_lc(vector)
        lc_frac_l.append(lc_frac)

        uc_frac, uc = single_uc(vector)
        uc_frac_l.append(uc_frac)

        inbound = inbound_ratio(vector, uc, lc)
        bound_frac.append(inbound)
    return [trans, black_frac, lc_frac_l, uc_frac_l, bound_frac]


def dtw(s, t):
    n, m = len(s), len(t)
    dtw_matrix = np.zeros((n + 1, m + 1))

    for i in range(n + 1):
        for j in range(m + 1):
            dtw_matrix[i, j] = np.inf

    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s[i - 1] is None:
                s[i - 1] = 0
            if t[j - 1] is None:
                t[j - 1] = 0
            cost = abs(s[i - 1] - t[j - 1])
            # take last min from a square box
            last_min = np.min([dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]])
            dtw_matrix[i, j] = cost + last_min

    return dtw_matrix


MYPATH = 'C:/Users/mg_gw/OneDrive/Dokumente/UNIBE 8. Semester/3_Pattern_Recognition/32_exercises/03_KWS/PatRec17_KWS_Data-master/PatRec17_KWS_Data-master/images/single_words_270/'


def feature_all_words(path):
    all_files = [f for f in listdir(path) if isfile(join(path, f))]  # extracts all filenames from folder

    ref_image = Image.open(path + all_files[0])

    imArray_compare = np.asarray(ref_image)
    compare_features = feature_extract(imArray_compare)

    for image in all_files[1:]:
        print(image)
        im = Image.open(path + image)
        imArray = np.asarray(im)

        features = feature_extract(imArray)
        score_list = []
        for i in range(len(features)):
            dtw_matrix = dtw(features[i], compare_features[i])
            score_list.append(dtw_matrix[-1, -1])
        print(score_list)

feature_all_words(MYPATH)



#align = dtw(black_frac46,black_frac2)



#print(len(black_frac))


