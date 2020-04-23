#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 17:29:59 2020

@author: anikajohn
"""


import numpy
from PIL import Image, ImageDraw

#######INSERT JPG FILE HERE 

# read image as RGB and add alpha (transparency)
im = Image.open("270.jpg").convert("RGBA") 

# convert to numpy (for convenience)
imArray = numpy.asarray(im)

import re

########CONVERT SVG TO TEXT FILE AND INSERT HERE 

file = open("270 copy.txt") #open text file 

lines = file.readlines() #each line becomes element in list lines 
file.close


for i in range(len(lines)): #iterate over all lines 
    
    pattern =  'L\s(\d+\.\d+)\s(\d+\.\d+)+' 
    line = lines[i]
    out = re.findall(pattern, line) #find pattern 
    
    if len(out) == 0:
        print("empty") #do nothing if the line does not contain pattern 
        
    else:
        
    
        out2 = list() #will contain float tuples 
    
        for tup in range(len(out)): #convert tuple strings in tuple floats 
        
            out2_0 = float(out[tup][0])
            out2_1 = float(out[tup][1])
            t2 = (out2_0,out2_1)
            
            out2.append(t2) #append float tuples 
        
        polygon = out2
        
        maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
        ImageDraw.Draw(maskIm).polygon(out2, outline=1, fill=1)
        mask = numpy.array(maskIm)
        
        # assemble new image (uint8: 0-255)
        newImArray = numpy.empty(imArray.shape,dtype='uint8')
        
        # colors (three first columns, RGB)
        newImArray[:,:,:3] = imArray[:,:,:3]
        
        # transparency (4th column)
        newImArray[:,:,3] = mask*255
        
        # back to Image from numpy
        newIm = Image.fromarray(newImArray, "RGBA")
        newIm.save(f'out{i}.png') #save word image to current directory 
        



# =============================================================================
# JUST NOTES/ TESTS DOWN HERE 
# =============================================================================


#
#for line in file:
   #print(line) 
#file.close



pattern =  'L\s(\d+\.\d+)\s(\d+\.\d+)+'
line = '<path fill="none" d="M 112.00 170.00 L 112.00 230.00 L 129.27 231.50 L 132.00 230.00 L 232.00 230.00 L 240.00 238.00 L 299.69 148.25 L 192.00 157.00 Z" stroke-width="1" id="270-01-01" stroke="black"/>"/>'
line2 = 'anika'

out = re.findall(pattern, line2)
#out2 = float(out)
print(type(out))

out2 = list()

for tup in range(len(out)):
    #print(float(out[tup][0]))
    out2_0 = float(out[tup][0])
    out2_1 = float(out[tup][1])
    t2 = (out2_0,out2_1)
    #print(type(t2))
    out2.append(t2)
    
print(out2)

# create mask
polygon = [(112.00,230.00),(129.27,231.50),(132.00,230.00),(232.00,230.00),(240.00,238.00),(299.69,148.25),(192.00,157.00)]
polygon2 = [(250.00,242.00),(250.19,248.19),(252.00,250.00),(412.00,250.00),(432.00,230.00),(452.00,249.00),(492.00,249.00),(511.00,230.00),(513.00,154.94),(366.72,145.00),(339.69,145.00),(299.69,148.25),(240.00,238.00 )]
maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
ImageDraw.Draw(maskIm).polygon(out2, outline=1, fill=1)
mask = numpy.array(maskIm)

# assemble new image (uint8: 0-255)
newImArray = numpy.empty(imArray.shape,dtype='uint8')

# colors (three first columns, RGB)
newImArray[:,:,:3] = imArray[:,:,:3]

# transparency (4th column)
newImArray[:,:,3] = mask*255

# back to Image from numpy
newIm = Image.fromarray(newImArray, "RGBA")
newIm.save("out4.png")

