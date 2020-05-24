# -*- coding: utf-8 -*-
"""
Created on Tue May 12 22:28:28 2020

@author: danie
"""


import os
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import re
import time

cwd_path = os.getcwd()




#lists all enrollment files
enr_dir = os.listdir(cwd_path + "/SignatureVerification/enrollment")

enrollment = list()

for i in enr_dir: 
    L = list()
    with open(cwd_path + "/SignatureVerification/enrollment/" + i) as enr:
        for j in enr:
            L.append(j.strip().split(" ")) 
    enrollment.append((i, L))

#lists all verification files
ver_dir = os.listdir(cwd_path + "/SignatureVerification/verification")

verification = list()

for i in ver_dir: 
    L = list()
    with open(cwd_path + "/SignatureVerification/verification/" + i) as ver:
        for j in ver:
            L.append(j.strip().split(" "))   
    verification.append((i, L))



# list of ground truth for verifications
ground_truth = list()
with open(cwd_path + "/SignatureVerification/gt.txt") as gt:
    for i in gt:
        ground_truth.append(i.strip())
        
ground_truth = [re.findall(r"\w$" ,i)[0] for i in ground_truth]



# list of people that wrote signatures
with open(cwd_path + "/SignatureVerification/users.txt") as us:
    users = list()
    for i in us:
        users.append(i.strip())




##############################################################################
#########################Features#############################################
##############################################################################

def get_features(signature):
    signature_feature_list = list()
    sig = np.asarray(signature)
    time = sig.take(0,axis=1)
    x = sig.take(1,axis=1)
    y = sig.take(2,axis=1)
    pressure = sig.take(3,axis=1)

    
    # calculates the velocity in x and y direction
    time_1 = np.array(np.append(np.array([np.inf]),time), float)
    time_2 = np.array(np.append(time, np.array([0])), float)
    delta_t = (time_2 - time_1)[:-1]    
         
    x_1 = np.array(np.append(np.array([0]),x), float)    
    x_2 = np.array(np.append(x, np.array([0])), float)
    delta_x = (x_2 - x_1)[:-1]    

    y_1 = np.array(np.append(np.array([0]),y), float)    
    y_2 = np.array(np.append(y, np.array([0])), float)
    delta_y = (y_2 - y_1)[:-1]     

    vx = (delta_x/delta_t)    
    
    vy = (delta_y/delta_t) 
    

    return x, y, vx, vy, pressure

##############################################################################
##############################################################################
##############################################################################


def DTW(vec_1, vec_2):
    dist = fastdtw(vec_1, vec_2, dist=euclidean)    
    return dist[0]
   
    
##############################################################################
##############################################################################
##############################################################################
  
    



count = 0
start_t = time.time()
t = 0
t_left = np.inf
for user in users:
    reference = [i for i in enrollment if re.match(f"^{user}.*",i[0])]
    print(user)
    DTW_x = list()
    DTW_y = list()
    DTW_vx = list()
    DTW_vy = list()
    DTW_pressure = list()
    
    validation_set = [i for i in verification if re.match(f"^{user}.*",i[0])]

    
    for index, ref in enumerate(reference):
        x_1, y_1, vx_1, vy_1, pressure_1 = get_features(ref[1])
        dtw_x = list()
        dtw_y = list()
        dtw_vx = list()
        dtw_vy = list()
        dtw_pressure = list()
        
        for i,j in enumerate(validation_set):
            x_2, y_2, vx_2, vy_2, pressure_2 = get_features(j[1])

            dist_x = DTW(x_1, x_2)
            dist_y = DTW(y_1, y_2)
            dist_vx = DTW(vx_1, vx_2)
            dist_vy = DTW(vy_1, vy_2)
            dist_pressure = DTW(pressure_1, pressure_2)
            count += 1
            if count % 100 == 0:
                t = (time.time()-start_t)/100
                t_left = t/(60*60)*(len(validation_set)*len(reference)*len(users)-count)
                start_t = time.time()
            print(count, " of ", len(verification)*len(reference), " and tot of : ", len(validation_set)*len(reference)*len(users), " time left [h] ", t_left)       
            
            dtw_x.append(dist_x)
            dtw_y.append(dist_y)
            dtw_vx.append(dist_vx)
            dtw_vy.append(dist_vy)
            dtw_pressure.append(dist_pressure)
        
        DTW_x.append(dtw_x) 
        DTW_y.append(dtw_y)  
        DTW_vx.append(dtw_vx)  
        DTW_vy.append(dtw_vy)  
        DTW_pressure.append(dtw_pressure)  
        
            
            
        print(DTW_pressure)       
     
    with open(f"Results/{user}_dist_x.txt", "w") as wr:
        wr.write(str(DTW_x))

    with open(f"Results/{user}_dist_y.txt", "w") as wr:
        wr.write(str(DTW_y))
        
    with open(f"Results/{user}_dist_vx.txt", "w") as wr:
        wr.write(str(DTW_vx))
        
    with open(f"Results/{user}_dist_vy.txt", "w") as wr:
        wr.write(str(DTW_vy))
        
    with open(f"Results/{user}_dist_pressure.txt", "w") as wr:
        wr.write(str(DTW_pressure))







