# -*- coding: utf-8 -*-
"""
Created on Wed May 13 11:20:12 2020

@author: danie
"""


import os
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import re
import copy
import matplotlib.pyplot as plt
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
        
ground_truth_g_f = [re.findall(r"\w$" ,i)[0] for i in ground_truth]
ground_truth_T_F = list(np.zeros(len(ground_truth_g_f)))
for i,j in enumerate(ground_truth_g_f):
    if j == "g":
        ground_truth_T_F[i] = True
    else:
        ground_truth_T_F[i] = False 





# list of people that wrote signatures
with open(cwd_path + "/SignatureVerification/users.txt") as us:
    users = list()
    for i in us:
        users.append(i.strip())
        
        
  
#==============================================================================
#normalization factor
Result_list = (os.listdir("Results/"))

all_list_pressure = list()
all_list_vx = list()
all_list_vy = list()
all_list_x = list()
all_list_y = list()

for user_index,user in enumerate(users):
    print("user: ", user)
    pattern = re.compile(f"^{user}.*")
    res = list(filter(lambda x: re.findall(f"^{user}",x), Result_list))
    if res != []:

        
        # seting up true_false vector for comparison
        user_indexes = list()
        for i,j in enumerate(ground_truth):
            if re.findall(f"^{user}",j) != []:
                user_indexes.append(i)
        
        user_ground__truth_T_F = list(np.zeros(len(user_indexes)))
        for i,j in enumerate(user_indexes):
            user_ground__truth_T_F[i] = ground_truth_T_F[j]
 
        with open(f"Results/{res[0]}") as data:
            dist_pressure = eval(data.read())
            #dist_pressure = list(dist_pressure/(np.max(dist_pressure)- np.min(dist_pressure)))

        with open(f"Results/{res[1]}") as data:
            dist_vx = eval(data.read())
            #dist_vx = list(dist_vx/(np.max(dist_vx)- np.min(dist_vx)))
        
        with open(f"Results/{res[2]}") as data:
            dist_vy = eval(data.read())
            #dist_vy = list(dist_vy/(np.max(dist_vy)- np.min(dist_vy)))
        
        with open(f"Results/{res[3]}") as data:
            dist_x = eval(data.read())
            #dist_x = list(dist_x/(np.max(dist_x)- np.min(dist_x)))
        
        with open(f"Results/{res[4]}") as data:
            dist_y = eval(data.read())
            #dist_y = list(dist_y/(np.max(dist_y)- np.min(dist_y)))
                

        
        all_list_pressure.append(dist_pressure) 
        all_list_vx.append(dist_vx)   
        all_list_vy.append(dist_vy)  
        all_list_x.append(dist_x)  
        all_list_y.append(dist_y)  
        

norm_factor_pressure = 1/(np.max(all_list_pressure)-np.min(all_list_pressure))
norm_factor_vx = 1/(np.max(all_list_vx)-np.min(all_list_vx))
norm_factor_vy = 1/(np.max(all_list_vy)-np.min(all_list_vy))
norm_factor_x = 1/(np.max(all_list_x)-np.min(all_list_x))
norm_factor_y = 1/(np.max(all_list_y)-np.min(all_list_y))


print(norm_factor_pressure, norm_factor_vx, norm_factor_vy,norm_factor_x,norm_factor_y )









#==============================================================================





      
  
Result_list = (os.listdir("Results/"))

for user in users:
    print("user: ", user)
    pattern = re.compile(f"^{user}.*")
    res = list(filter(lambda x: re.findall(f"^{user}",x), Result_list))
    if res != []:

        
        # seting up true_false vector for comparison
        user_indexes = list()
        for i,j in enumerate(ground_truth):
            if re.findall(f"^{user}",j) != []:
                user_indexes.append(i)
        
        user_ground__truth_T_F = list(np.zeros(len(user_indexes)))
        for i,j in enumerate(user_indexes):
            user_ground__truth_T_F[i] = ground_truth_T_F[j]
            

        
        
        
        
        
        
        
        
        
        
        
        with open(f"Results/{res[0]}") as data:
            dist_pressure = eval(data.read())
            dist_pressure = list(np.asarray(dist_pressure)*norm_factor_pressure)

        with open(f"Results/{res[1]}") as data:
            dist_vx = eval(data.read())
            dist_vx = list(np.asarray(dist_vx)*norm_factor_vx)
        
        with open(f"Results/{res[2]}") as data:
            dist_vy = eval(data.read())
            dist_vy = list(np.asarray(dist_vy)*norm_factor_vy)
        
        with open(f"Results/{res[3]}") as data:
            dist_x = eval(data.read())
            dist_x = list(np.asarray(dist_x)*norm_factor_x)
        
        with open(f"Results/{res[4]}") as data:
            dist_y = eval(data.read())
            dist_y = list(np.asarray(dist_y)*norm_factor_y)
                
        correct = 0
        false = 0
        #print(dist_pressure[0])
        #print(dist_vx[0])    
        #print(dist_vy[0])
        #print(dist_x[0])
        #print(dist_y[0])
        #print(user_ground__truth_T_F)
        threshoolds = np.asarray(range(0, 100))/100
        precision_racall_list = list([[],[]])
        
        precision_racall_list_pressure = list([[],[]])
        precision_racall_list_vx = list([[],[]])
        precision_racall_list_vy = list([[],[]])
        precision_racall_list_x = list([[],[]])
        precision_racall_list_y = list([[],[]])
        threshoold_list = list()                                
        
        

        for threshoold in threshoolds:

            
            #print(threshoold)
    ########calculation of precision and recall for all different features for one user      
            pressure_all_ref_precision = list()
            pressure_all_ref_recall = list()
            for index_reference in range(len(dist_pressure)):  
                dist_true_wrong = list(dist_pressure[index_reference] < threshoold)
                T_F_precision = np.logical_and((np.asarray(dist_true_wrong) == True) , (np.asarray(user_ground__truth_T_F) == True))
                if sum(dist_true_wrong)==0:
                    pressure_all_ref_precision.append(1)    
                else:
                    pressure_all_ref_precision.append(sum(T_F_precision)/sum(dist_true_wrong))
                
                T_F_recall = np.logical_and((np.asarray(dist_true_wrong) == True) , (np.asarray(user_ground__truth_T_F) == True))
                
                pressure_all_ref_recall.append(sum(T_F_recall)/sum(user_ground__truth_T_F))
            
     
    
            vx_all_ref_precision = list()
            vx_all_ref_recall = list()
            for index_reference in range(len(dist_vx)):  
                dist_true_wrong = list(dist_vx[index_reference] < threshoold)
                T_F_precision = np.logical_and((np.asarray(dist_true_wrong) == True) , (np.asarray(user_ground__truth_T_F) == True))
                if sum(dist_true_wrong)==0:
                    vx_all_ref_precision.append(1)    
                else:
                    vx_all_ref_precision.append(sum(T_F_precision)/sum(dist_true_wrong))
                
                
                T_F_recall = np.logical_and((np.asarray(dist_true_wrong) == True) , (np.asarray(user_ground__truth_T_F) == True))
                
                vx_all_ref_recall.append(sum(T_F_recall)/sum(user_ground__truth_T_F))
    
    
    
            vy_all_ref_precision = list()
            vy_all_ref_recall = list()
            for index_reference in range(len(dist_vy)):  
                dist_true_wrong = list(dist_vy[index_reference] < threshoold)
                T_F_precision = np.logical_and((np.asarray(dist_true_wrong) == True) , (np.asarray(user_ground__truth_T_F) == True))
                if sum(dist_true_wrong)==0:
                    vy_all_ref_precision.append(1)    
                else:
                    vy_all_ref_precision.append(sum(T_F_precision)/sum(dist_true_wrong))
                
                T_F_recall = np.logical_and((np.asarray(dist_true_wrong) == True) , (np.asarray(user_ground__truth_T_F) == True))
                
                vy_all_ref_recall.append(sum(T_F_recall)/sum(user_ground__truth_T_F))
           
            
            
    
            x_all_ref_precision = list()
            x_all_ref_recall = list()
            for index_reference in range(len(dist_x)):  
                dist_true_wrong = list(dist_x[index_reference] < threshoold)
                T_F_precision = np.logical_and((np.asarray(dist_true_wrong) == True) , (np.asarray(user_ground__truth_T_F) == True))
                if sum(dist_true_wrong)==0:
                    x_all_ref_precision.append(1)    
                else:
                    x_all_ref_precision.append(sum(T_F_precision)/sum(dist_true_wrong))
                
                T_F_recall = np.logical_and((np.asarray(dist_true_wrong) == True) , (np.asarray(user_ground__truth_T_F) == True))
                
                x_all_ref_recall.append(sum(T_F_recall)/sum(user_ground__truth_T_F))
    
    
            y_all_ref_precision = list()
            y_all_ref_recall = list()
            for index_reference in range(len(dist_y)):  
                dist_true_wrong = list(dist_y[index_reference] < threshoold)
                T_F_precision = np.logical_and((np.asarray(dist_true_wrong) == True) , (np.asarray(user_ground__truth_T_F) == True))
                if sum(dist_true_wrong)==0:
                    y_all_ref_precision.append(1)    
                else:
                    y_all_ref_precision.append(sum(T_F_precision)/sum(dist_true_wrong))
                
                T_F_recall = np.logical_and((np.asarray(dist_true_wrong) == True) , (np.asarray(user_ground__truth_T_F) == True))
                
                y_all_ref_recall.append(sum(T_F_recall)/sum(user_ground__truth_T_F))
    
            threshoold_list.append(threshoold)
    
    
            precision_racall_list_pressure
            # only for preassure
            precision_racall_list_pressure[0].append(np.mean(pressure_all_ref_precision))
            precision_racall_list_pressure[1].append(np.mean(pressure_all_ref_recall))
            
            precision_racall_list_vx[0].append(np.mean(vx_all_ref_precision))
            precision_racall_list_vx[1].append(np.mean(vx_all_ref_recall))

            precision_racall_list_vy[0].append(np.mean(vy_all_ref_precision))
            precision_racall_list_vy[1].append(np.mean(vy_all_ref_recall))
    
            precision_racall_list_x[0].append(np.mean(x_all_ref_precision))
            precision_racall_list_x[1].append(np.mean(x_all_ref_recall))
            
            precision_racall_list_y[0].append(np.mean(y_all_ref_precision))
            precision_racall_list_y[1].append(np.mean(y_all_ref_recall))
            
            
            
            
            
            
            # for all features
            tot_mean_precision = np.mean([np.mean(pressure_all_ref_precision),np.mean(vx_all_ref_precision),np.mean(vy_all_ref_precision),np.mean(x_all_ref_precision),np.mean(y_all_ref_precision)])
            tot_mean_recall = np.mean([np.mean(pressure_all_ref_recall),np.mean(vx_all_ref_recall),np.mean(vy_all_ref_recall),np.mean(x_all_ref_recall),np.mean(y_all_ref_recall)])
            
            precision_racall_list[0].append(tot_mean_precision)
            precision_racall_list[1].append(tot_mean_recall)

                                
        #print(precision_racall_list)
  
        plt.plot(precision_racall_list[1],precision_racall_list[0])
        #plt.show()
"""        
        plt.plot(precision_racall_list_pressure[1],precision_racall_list_pressure[0], marker = "o")
        #plt.show()
       
        plt.plot(precision_racall_list_vx[1],precision_racall_list_vx[0], marker = "o")
        #plt.show()       
        
        plt.plot(precision_racall_list_vy[1],precision_racall_list_vy[0], marker = "o")
        #plt.show()       
       
        plt.plot(precision_racall_list_x[1],precision_racall_list_x[0], marker = "o")
        #plt.show()       
      
        plt.plot(precision_racall_list_y[1],precision_racall_list_y[0], marker = "o")
        plt.legend(["tot","pressure","vx","vy","x","y"])

        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.savefig(f"plots_all_features/{user}_precision_recall.png")
        plt.show()       

        plt.plot(precision_racall_list[1],threshoold_list, marker = "o")
        plt.plot(precision_racall_list[0],threshoold_list, marker = "o")
        plt.legend(["recall","presicion"])
        plt.xlabel("threshoold")
        plt.ylabel("precision / recall")
        plt.savefig(f"plots_all_features/{user}_precision-recall_threshoold.png")
        plt.show()
        
        
        
"""        
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(["user_1","user_2","user_3","user_4","user_5","user_6","user_7","user_8","user_9","user_10",
            "user_11","user_12","user_13","user_14","user_15","user_16","user_17","user_18","user_19","user_20",
            "user_21","user_22","user_23","user_24","user_25","user_26","user_27","user_28","user_29","user_30"], ncol= 2, fontsize = "xx-small")
plt.savefig(f"plots_all_features/all_users_precision_recall.png")
plt.show()   

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
