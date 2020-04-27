# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 20:39:10 2020

@author: daniel
"""

import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import copy
from shapely.geometry import Polygon, Point
import re
import numpy as np


# start: read in data ---------------------------------------------------------

path = os.getcwd()+"/PatRec17_KWS_Data-master\ground-truth\locations"
ground_truth = list()
for i in os.listdir(path):
    ground_truth.append([])
    paths = os.getcwd()+"/PatRec17_KWS_Data-master\ground-truth\locations/"+i
    with open(paths, "r") as data:  
        for i in data:
            ground_truth[len(ground_truth)-1].append(i.split(","))   
        data.close()



path = os.getcwd()+"/PatRec17_KWS_Data-master/images/images_bw"
images = list()
for i in os.listdir(path):
    paths = os.getcwd()+"/PatRec17_KWS_Data-master\images/images_bw/"+i
    images.append(copy.deepcopy(np.asarray(Image.open(paths).convert("RGBA"))))





for i,j in enumerate(ground_truth):
    current_image = images[i]
    
    new_folder = os.getcwd() + f"/for_image_{i}"
   
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)
    
    for z,p in enumerate(j):

        polygon_cordinated = re.findall(r" +\d+\.\d+", str(p))
        
        
        
        


        
        
        
        
        if len(polygon_cordinated) == 0:
            pass
        else:
    
           
            
            
            poly = list()
            tup = list()
            for i,c in enumerate(polygon_cordinated):
                if(i > 0 and i%2 == 0):
                    poly.append(tuple(tup))
                    tup = list()
                tup.append(eval(c.strip()))
            poly.append(tuple(tup))
            

            
            #polygon cut
            
            mask_img = Image.new("L" , (current_image.shape[1],current_image.shape[0]), 0)
            ImageDraw.Draw(mask_img).polygon(poly, outline = 1, fill = 1)
            mask = np.asarray(mask_img)
            
            

            if len(current_image.shape) == 3:
                current_image = current_image[:,:,0]
            
            cut_img = np.empty(current_image.shape)
            

            

            
            
            

            
            cut_img[:,:] = current_image[:,:]*mask
            
            
            
            #Image.fromarray(cut_img).show()
    
            indexes = np.where(mask)
            
            


            print(cut_img)
            
            
            
            if len(indexes[0]) == 0:
                pass
            else:
            
                row_max = np.max(indexes[0])
                row_min = np.min(indexes[0])
                
                col_max = np.max(indexes[1])
                col_min = np.min(indexes[1])
                
               
                
                cut_img = Image.fromarray(cut_img).crop((col_min,row_min, col_max,row_max))
                
               # cut_img.show()
                
                cut_img = cut_img.convert("L")
                
                cut_img.save(new_folder + f"/word_{z}.png", "png")
            
            
            
            
            
            """
            cut_image = np.zeros((images[0].shape[0],images[0].shape[1]))+ 255
            count = 0
            for i,j in np.ndenumerate(images[0]):
                row, col = i
                
                if (row%100 == 0) and (col %10000 == 0):
                 #   print(row)
                    a=5    
                
                if Polygon(poly).contains(Point(col,row)):
                    cut_image[row][col] = images[0][row][col]
                    count += 1
                    
            cut_image[cut_image <= 10] = 0
            cut_image[cut_image > 10] = 255
            
            cut_cut_image = cut_image[cut_image < 255]
            print(cut_cut_image)
            Image.fromarray(cut_image).show()
            
            """


# end: read in data -----------------------------------------------------------