# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:26:43 2020

@author: Admin
"""


import matplotlib.pyplot as plt
from skimage import filters
from os import listdir
from os.path import isfile, join
from PIL import Image, ImageDraw
import numpy as np





# I imported the image like this, just for reference
img = color.rgb2gray(io.imread('out3_cropped.png'))
img = np.asarray(img)


# the fractions that are in the feature vector are considered from the bottom
# pixel in the bottom left = pixel number 0 

# so that if we plot it the outline looks like the word


# the functions return a list, but the output format can easily be changed



# =============================================================================
# Upper contour
# =============================================================================

def get_uc(img):
    # convert the floats to integers or else = is not really 0 but 0.0000....something
    img.astype(int)  
    
    
    # if there are different heights would have to extract the height
    # in the first for loop
    
    height, length = img.shape  #height = number of rows, length = number of cols
    

    uc_list= [ ]

    counter = 0
    
    for col in range(0,length):  

        height = len(img[:,col])
        for row in range(0,height): 

            if img[row, col] == 0:
                fraction = row/height
                
                uc_list.append(1-fraction)   
                break
            elif row == (height-1):
                
                #print('None')
                
                # appends None to the list if there is no black pixels
                # could change None to anything you want to have 
                # if there is no black pixel in that column
                uc_list.append(None)
                
    ### change output format here if needed ###

    return uc_list
                
   
### To plot the contour****                 
# a = get_uc(img)

# b = [i for i in range(img.shape[1])]



# plt.plot(b,a)


# =============================================================================
# Lower contour
# =============================================================================
  
    
def get_lc(img):
    # convert the floats to integers or else = is not really 0 but 0.0000....something
    img.astype(int)  
    
    
    # if there are different heights would have to extract the height
    # in the first for loop
    
    height, length = img.shape  #height = number of rows, length = number of cols
    

    lc_list= [ ]

    counter = 0
    
    for col in range(0,length):  

        height = len(img[:,col])
        for row in reversed(range(0,height)): 

            if img[row, col] == 0:
                fraction = row/height
                #print(fraction)
                lc_list.append(1-fraction)   
                break
            elif row == (0):
                
                
                
                # appends None to the list if there is no black pixels
                # could change None to anything you want to have 
                # if there is no black pixel in that column
                lc_list.append(None)

    return lc_list
   
### plot lower  contour ###
                 
# a = get_lc(img)

# b = [i for i in range(img.shape[1])]

# plt.plot(b,a)
# plt.ylim(0,1)


