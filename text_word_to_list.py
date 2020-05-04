# -*- coding: utf-8 -*-
"""
Created on Mon May  4 09:55:46 2020

@author: Admin
"""

import numpy
from PIL import Image, ImageDraw



import re

########CONVERT SVG TO TEXT FILE AND INSERT HERE 

file = open("transcription.txt") #open text file 

lines = file.readlines() #each line becomes element in list lines 
file.close


n = 35  # number of pages
transcription_list = []


page_number = 270
#page_number = '270'
a = 0

for i in range(len(lines)): #iterate over all lines 

    
        pattern =  r'^(\d{3})-\d{2}-\d{2}\s(.+)\n' 
        line = lines[i]
        #print(line)
        
        
        
        pattern1 =  '^(\d{3})'
        page_numb = re.findall(pattern1, line) #find pattern 
        page_numb = str(page_numb)
        page_numb = page_numb.strip(' \'[]')
        #print('g',page_numb)
        page_numb = int(page_numb)
        
        index = page_numb-270
        
        #print(index)

        index = page_numb-270
        
        #print(index)
        
        
        out1 = re.search(pattern, line).group(1) #find pattern 
        
        out1 = str(out1)
            
        out1 = out1.strip(' \'[]')
        
        out1 = int(out1)
        
        out2 = re.search(pattern, line).group(2) #find pattern 
        
        out2 = str(out2)
            
        out2 = out2.strip(' \'[]')
        
        out = tuple((out1, out2))
        
        
        #out = float(out)
        
        
        transcription_list.append(out)
        
        
        
           
            
            
 # all words = 3726
print(len((transcription_list)))
print((transcription_list))
#print(transcription_list[1])


print(transcription_list[4][1])

page_270 = []
for i in range(len(transcription_list)):
    if transcription_list[i][0] == 270:
        #print(transcription_list[i][1])
        page_270.append(transcription_list[i][1])

print(len(page_270))
print(page_270)   