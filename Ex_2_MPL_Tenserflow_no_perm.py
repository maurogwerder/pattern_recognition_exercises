import numpy as np
import time
import random
import math
import copy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import random

def build_model(neurons):
    model = tf.keras.Sequential() #builds model based on fully connected layers
    
    
    
    
    model.add(layers.Dense(neurons, activation ="relu", input_shape=(784,)))      #intermediat layers based in relu activation input
    model.add(layers.Dense(10, activation ="softmax"))    # output layer based on softmax activation shape (10,)
    
    model.summary()
 
    return model


def train_network(model, train_data,train_label,learning_rate, test_data = None , test_label = None, Epochs = 20):
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate),
                  metrics=['accuracy'])
    
    model.fit(train_data, train_label, epochs = Epochs, batch_size=128)
    score_train = model.evaluate(train_data, train_label)
    score_test = model.evaluate(test_data, test_label)
    print(score_train,score_test)
    return (model, score_test, score_train)






def main():
   
    start_t = time.time()
    train = open(r"mnist_train.csv","r")
    train_set_data = list()
    train_set_label = list()
   
    for i in train: 
        train_set_data.append(list(map(int,(i.strip().split(","))))[1:])
        label = list(map(int,(i.strip().split(","))))[0]
        lab = list()
        for j in range(0,10):
            if label == j:
                lab.append(1)
            else:
                lab.append(0)
        train_set_label.append(lab)
  
    train.close()
    train_set_data_np = np.asarray(train_set_data)
    train_set_data_np = train_set_data_np/255
    train_set_label_np = np.asarray(train_set_label)
    
    test = open(r"mnist_test.csv","r")
    test_set_data = list()
    test_set_label = list()


    for i in test: 
        test_set_data.append(list(map(float,(i.strip().split(","))))[1:])
        label = list(map(int,(i.strip().split(","))))[0]
        lab = list()
        for j in range(0,10):
            if label == j:
                lab.append(1)
            else:
                lab.append(0)
        test_set_label.append(lab)
    

    
    test.close()
    test_set_data_np = np.asarray(test_set_data)
    test_set_data_np = test_set_data_np/255
    test_set_label_np = np.asarray(test_set_label)
 
    
    
    #data permutation start---------------------------------------------------   


    """
        
    train_set_data_perm = copy.deepcopy(train_set_data)
    for i,j in enumerate(train_set_data):
        train_set_data_perm[i] = random.sample(j, 784)
        
        
        #img = np.asarray(train_set_data[i]).reshape((28,28))
        #plt.imshow(img) 
        #plt.show() 

        #img = np.asarray(train_set_data_perm[i]).reshape((28,28))
        #plt.imshow(img) 
        #plt.show()        
        


    test_set_data_perm = copy.deepcopy(test_set_data)
    for i,j in enumerate(test_set_data):
        test_set_data_perm[i] = random.sample(j, 784)
        
           
        #img = np.asarray(test_set_data[i]).reshape((28,28))
        #plt.imshow(img) 
        #plt.show() 

        #img = np.asarray(test_set_data_perm[i]).reshape((28,28))
        #plt.imshow(img) 
        #plt.show()        
        


    """

        
      
        
    #data permutation end-----------------------------------------------------    
      
    
    
    print("reading in Data finished after ", time.time()-start_t, " seconds")
    print( train_set_data_np.shape)
    print( test_set_data_np.shape)
    
    best_learning_rate = list()
    for i in [0.00001, 0.0001,0.001, 0.01, 0.1, 1]:
        neural_network = build_model(100)
        result = train_network(neural_network,train_set_data_np,train_set_label_np,i,test_set_data_np,test_set_label_np, 20)
        best_learning_rate.append((result[1][1], i))
        Results = open("results.txt", "a") 
        Results.write("Testing Network: "+ "learning rate = "+ str(i) + " accurancy = "+ str(result[1][1]) + " number of neurons = "+ str(100) + " epochs size = "+ str(20) + " batch size of 128" + "\n")
        Results.close()
    best_lr = max(best_learning_rate)
    print("best learning rate is = ", best_lr[1], " whit an accurancy of: ", best_lr[0])    
    best_layer_size = list()
    for j in [10, 20, 30, 40, 50,60,70,80,90, 100]:
        neural_network = build_model(j)  
        result = train_network(neural_network,train_set_data_np,train_set_label_np,best_lr[1],test_set_data_np,test_set_label_np, 20)
        best_layer_size.append((result[1][1], j))
        Results = open("results.txt", "a") 
        Results.write("Testing Network: "+ "learning rate = "+ str(best_lr[1]) + " accurancy = "+ str(result[1][1]) + " number of neurons = "+ str(j) + " epochs size = "+ str(20) + " batch size of 128" + "\n")
        Results.close()
    best_ls = max(best_layer_size)  
    print("best layer size is = ", best_ls[1], " whit an accurancy of: ", best_ls[0])    
    accurancy_train_set = list()
    accurancy_test_set = list()
    best_epoch_size = list()
    batches = list()
    batch = 0
    neural_network = build_model(best_ls[1])
    for e in range(1, 1000):
        batch = e*1
        print("batch = ", batch)
        result = train_network(neural_network,train_set_data_np,train_set_label_np,best_lr[1],test_set_data_np,test_set_label_np, 1)
        accurancy_train_set.append(result[2][1])
        accurancy_test_set.append(result[1][1])
        best_epoch_size.append((result[1][1], batch))
        batches.append(batch)
        Results = open("results.txt", "a") 
        Results.write("Testing Network: "+ "learning rate = "+ str(best_lr[1]) + " accurancy = "+ str(result[1][1]) + " number of neurons = "+ str(best_ls[1]) + " epochs size = "+ str(e) + " batch size of 128" + "\n")
        Results.close()
        neural_network = result[0]
        
        plt.plot(batches, accurancy_train_set); plt.title("learning rate: " +str(best_lr[1])+ "n-size: "+str(best_ls[1]))
        plt.plot(batches, accurancy_test_set); plt.legend(["train","test"])
        plt.savefig((str("learning rate_" +str(best_lr[1])+ "_n-size_"+str(best_ls[1])+"_4.png")), format="png")
        plt.show()
        
    best_ep = max( best_epoch_size)
    print("best epoch size is = ", best_ep[1], " whit an accurancy of: ", best_ep[0])
    plt.plot(batches, accurancy_train_set); plt.title("learning rate: " +str(best_lr[1])+ "n-size: "+str(best_ls[1]))
    plt.plot(batches, accurancy_test_set); plt.legend(["train","test"])
    plt.savefig((str("learning rate_" +str(best_lr[1])+ "_n-size_"+str(best_ls[1])+"_4.png")),format="png")
    plt.show()
    
    
    Results = open("results.txt", "a") 
    Results.write("best Network: "+ "best learning rate = "+ str(best_lr[1]) + " best number of neurons = "+ str(best_ls[1]) + " best epochs size = "+ str(best_ep[1]) + " batch size of 128" + "\n")
    Results.close()
    
    
    
    
    
    
    
    
    
    
    
    
    return 0






main()
