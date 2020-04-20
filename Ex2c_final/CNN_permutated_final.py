"""
CNN with 3 conv layers and a fully connected classification layer
PATTERN RECOGNITION EXERCISE:
Fix the three lines below marked with PR_FILL_HERE


CNN -> Convolution neural network normaly consist of multiple convolutional layers followd
with a pooling layer (mostly max_pooling). afterwards fully connected layers
advantages to MLP are:
    -2D, 3D Neurol construct
    -shared weights
    -lokal conectivity



"""
import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
import copy
import random

class Flatten(nn.Module):
    """
    Flatten a convolution block into a simple vector.

    Replaces the flattening line (view) often found into forward() methods of networks. This makes it
    easier to navigate the network with introspection
    """
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class PR_CNN(nn.Module):
    """
    Simple feed forward convolutional neural network

    Attributes
    ----------
    expected_input_size : tuple(int,int)
        Expected input size (width, height)
    conv1 : torch.nn.Sequential
    conv2 : torch.nn.Sequential
    conv3 : torch.nn.Sequential
        Convolutional layers of the network
    fc : torch.nn.Linear
        Final classification fully connected layer

    """

    def __init__(self, **kwargs):
        """
        Creates an CNN_basic model from the scratch.

        Parameters
        ----------
        output_channels : int
            Number of neurons in the last layer
        input_channels : int
            Dimensionality of the input, typically 3 for RGB
        """
        super(PR_CNN, self).__init__()

        # PR_FILL_HERE: Here you have to put the expected input size in terms of width and height of your input image
        self.expected_input_size = (28,28)

        # First layer
        self.conv1 = nn.Sequential(
            # PR_FILL_HERE: Here you have to put the input channels, output channels ands the kernel size
            nn.Conv2d(in_channels=1, out_channels=24, kernel_size=5, stride=3),
            nn.LeakyReLU()
        )

        # Classification layer
        self.fc = nn.Sequential(
            Flatten(),
            # PR_FILL_HERE: Here you have to put the output size of the linear layer. DO NOT change 1536!
            nn.Linear(1536, 10)
        )

    def forward(self, x):
        """
        Computes forward pass on the network

        Parameters
        ----------
        x : Variable
            Sample to run forward pass on. (input to the model)

        Returns
        -------
        Variable
            Activations of the fully connected layer
        """
        x = self.conv1(x)
        x = self.fc(x)
        return x



def training(net, train_data,labels, learning_rate, epochs):

    net = net

    optimizer = torch.optim.SGD(net.parameters() , lr = learning_rate)
    Log_sof = nn.LogSoftmax(dim = 1)
    loss = nn.NLLLoss()
    
    epochs = epochs
    data_size = train_data.shape[0]
    
    Loss_sum_sum = 0
    
    for e in range(0, epochs):
        Loss_sum = 0
        print("epoch : ", e)
        
        f = 0
        c = 0
        
        for i,j in enumerate(train_data):
            Input = torch.Tensor(j[np.newaxis,np.newaxis,:,:])
            Label= np.argmax(labels[i])
            output = ((net.forward(Input)))
            optimizer.zero_grad()
    
            
            Estimate = np.argmax(output.detach().numpy())
            Estimate = np.argmax(output.detach().numpy())
            if Estimate == Label:
                c += 1
            else:
                f +=1  
            
            """
            if( (f+c) % 1000 == 0):
                print( (c+f)/data_size*100 , " % donne ", "      with "  , c/(c+f)*100, " % correct ")
            """   
            
            
            o = output
    
            l = torch.tensor([Label], dtype = torch.long)
      
    
            soft_m = Log_sof(output)
    
            back = loss(soft_m, l)
            
            #print(back)
            
            back.backward()
            
            
            Loss_sum += back.detach().numpy()
            
        
            
            
            optimizer.step()
            

        Loss_sum_sum += Loss_sum/len(train_data)  
            
            
        print("train: ", (c+f)/data_size*100 , " % donne ", "      with "  , c/(c+f)*100, " % correct ", "loss = ", Loss_sum/len(train_data))
            
            
            
    return (net,c/(c+f),Loss_sum_sum/((epochs)))




def testing(net, test_data, labels):

    net = net

    
    epochs = 1
    
    
    Log_sof = nn.LogSoftmax(dim = 1)
    loss = nn.NLLLoss()

    
    Loss_sum_sum = 0
    
    data_size = test_data.shape[0]
    for e in range(0, epochs):
        

        print("epoch : ", e)
        
        f = 0
        c = 0
        Loss_sum = 0
        for i,j in enumerate(test_data):
            
            Input = torch.Tensor(j[np.newaxis,np.newaxis,:,:])
            Label= np.argmax(labels[i])
            output = ((net.forward(Input)))
    
            
            Estimate = np.argmax(output.detach().numpy())
            Estimate = np.argmax(output.detach().numpy())
            if Estimate == Label:
                c += 1
            else:
                f +=1  
            
            """
            if( (f+c) % 1000 == 0):
                print( (c+f)/data_size*100 , " % donne ", "      with "  , c/(c+f)*100, " % correct ")
            """    
            
            
            o = output
    
            l = torch.tensor([Label], dtype = torch.long)
      
   
        
            soft_m = Log_sof(output)
    
            back = loss(soft_m, l)
            back.backward()
            
            
            Loss_sum += back.detach().numpy()
            
        Loss_sum_sum += Loss_sum/len(test_data)
            
            
            
        print("test: ", (c+f)/data_size*100 , " % donne ", "      with "  , c/(c+f)*100, " % correct ", "loss = ", Loss_sum_sum/(epochs))
        
        
        return (net,c/(c+f),Loss_sum_sum/epochs)
            
            
            
            
            
            
            
        
      
            






























def main():
   
    start_t = time.time()
    train = open(r"mnist_train.csv","r")
    train_set_data = list()
    train_set_label = list()
   
    reduced_set = "no"
    
    if reduced_set == "yes":
        
 #-----------------------------------------------------------------------------   
        
        for i in train: 
            if len(train_set_data) < 1000:   
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
        train_set_data_picture = list()

        
        train_set_data_np = train_set_data_np
        train_set_label_np = np.asarray(train_set_label)
  
#-----------------------------------------------------------------------------   

      
        test = open(r"mnist_test.csv","r")
        test_set_data = list()
        test_set_label = list()
    
        
        
        for i in test:
            if len(test_set_data) < 1000:   
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
        test_set_data_picture = list()

        test_set_data_np = test_set_data_np
        test_set_label_np = np.asarray(test_set_label)
        
#-----------------------------------------------------------------------------           
 
    else:
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
        train_set_data_picture = list()

        
        train_set_data_np = train_set_data_np
        train_set_label_np = np.asarray(train_set_label)
  
#-----------------------------------------------------------------------------           
        
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
        test_set_data_picture = list()

        test_set_data_np = test_set_data_np
        test_set_label_np = np.asarray(test_set_label)
        
    print("reading in Data finished after ", time.time()-start_t, " seconds")
    print( train_set_data_np.shape)
    print( test_set_data_np.shape)

    









#data permutation start---------------------------------------------------   


    
        
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
        

    train_set_data_perm_np = np.asarray(train_set_data_perm)
    train_set_data_np = train_set_data_perm_np
    
    test_set_data_perm_np = np.asarray(test_set_data_perm)
    test_set_data_np = test_set_data_perm_np

    for i in range(0, len(train_set_data_np)):
            train_set_data_picture.append(train_set_data_np[i].reshape(28,28))    


    for i in range(0, len(test_set_data_np)):
            test_set_data_picture.append(test_set_data_np[i].reshape(28,28))


    train_set_data_picture = np.asarray(train_set_data_picture)
    test_set_data_picture = np.asarray(test_set_data_picture)
    
    print(train_set_data_picture.shape)

      
        
    #data permutation end-----------------------------------------------------  



































    
#-----------------------------------------------------------------------------
    #optimize learning rate
    
    
    
    best_learning_rate = list()
    for i in [0.1,0.01,0.001,0.0005,0.0001,0.00005,0.00001]:
        Network = PR_CNN()
        
        Train_result = training(Network, train_set_data_picture ,train_set_label, i, 10)
        Network = Train_result[0]
        Test_result = testing(Network, test_set_data_picture, test_set_label)
        
        
        best_learning_rate.append((Test_result[1], i))
        Results = open("CNN_results.txt", "a") 
        Results.write("CNN : Testing Network: "+ "learning rate = "+ str(i) + " accurancy = "+ str(Test_result[1]) + " epochs size = "+ str(10) + "\n")
        Results.close()
    best_lr = max(best_learning_rate)
    print("best learning rate is = ", best_lr[1], " whit an accurancy of: ", best_lr[0])


    

#-----------------------------------------------------------------------------
    
    Network = PR_CNN()
    

    
    
    
    # optimize epoch size
    accurancy_train_set = list([[],[]])
    accurancy_test_set = list([[],[]])
    best_epoch_size = list()
    batches = list()  
    
    
    
    
    
    
    
    for e in range(1, 150):
        
        
        
        Train_result = training(Network, train_set_data_picture ,train_set_label, best_lr[1], 1)
        
        Test_result = testing(Network, test_set_data_picture, test_set_label)

        accurancy_train_set[0].append(Train_result[1])
        accurancy_test_set[0].append(Test_result[1])
        accurancy_train_set[1].append(Train_result[2])
        accurancy_test_set[1].append(Test_result[2])
        best_epoch_size.append((Test_result[1],Test_result[2], e))
        batches.append(e)
        Results = open("CNN_results.txt", "a") 
        Results.write("Testing Network: "+ "learning rate = "+ str(best_lr[1]) + " accurancy = "+ str(Test_result[1]) + " epochs size = "+ str(e) + "\n")
        Results.close()
        
        Res = open("CNN_for_plot_accurancies.txt", "w") 
        Res.write(str(batches) +"\n" +str(accurancy_train_set[0]) + " \n" + str(accurancy_test_set[0]))
        Res.close()

        Res = open("CNN_for_plot_Loss.txt", "w") 
        Res.write(str(batches) +"\n" +str(accurancy_train_set[1]) + " \n" + str(accurancy_test_set[1]))
        Res.close()

    
        Network = Train_result[0]
        
        plt.plot(batches, accurancy_train_set[0]); plt.title("learning rate: " +str(best_lr[1]));plt.xlabel("epochs");plt.ylabel("accurancy")
        plt.plot(batches, accurancy_test_set[0]); plt.legend(["train","test"])
        plt.savefig(str("accurancy:_learning rate_" +str(best_lr[1])+ "_4.png"), format="png")
        plt.show()
        
        plt.plot(batches, accurancy_train_set[1]); plt.title("learning rate: " +str(best_lr[1]))
        plt.plot(batches, accurancy_test_set[1]); plt.legend(["train","test"])
        plt.xlabel("epochs")
        plt.ylabel("Loss")
        plt.savefig(str("Loss:_learning rate_" + str(best_lr[1]) + "_4.png"), format="png")
        plt.show()


    best_ep = max( best_epoch_size)
    print("best epoch size is = ", best_ep[2], " whit an accurancy of: ", best_ep[0])
   
    plt.plot(batches, accurancy_train_set[0]); plt.title("learning rate: " +str(best_lr[1]))
    plt.plot(batches, accurancy_test_set[0]); plt.legend(["train","test"])
    plt.xlabel("epochs")
    plt.ylabel("accurancy")
    plt.savefig(str("accurancy:_learning rate_" +str(best_lr[1])+ "_4.png"),format="png")
    plt.show()



    plt.plot(batches, accurancy_train_set[1]); plt.title("learning rate: " +str(best_lr[1]))
    plt.plot(batches, accurancy_test_set[1]); plt.legend(["train","test"])
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig(str("Loss:_learning rate_" +str(best_lr[1])+ "_4.png"),format="png")
    plt.show()

    Results = open("CNN_results.txt", "a") 
    Results.write("best epoch size is = " + str(best_ep[1]) + " whit an accurancy of: " + str(best_ep[0]) + "\n")
    Results.close()

    print("testing multiple networks")


    


   

    #choose best network of 10 initialisations--------------------------------
    
    best_net = list()
    for i in range(0, 10):
        
        Network = PR_CNN()
        copy_net = copy.deepcopy(Network)
        Train_result = training(Network, train_set_data_picture ,train_set_label, best_lr[1], best_ep[2])     
        Test_result = testing(Network, test_set_data_picture, test_set_label)
        best_net.append((Test_result[1],Test_result[2],copy_net))
    
        print("net_" , i , "with accurancy: ", Test_result[1], Test_result[2], Train_result[1], Train_result[2])
        
        Results = open("CNN_results.txt", "a") 
        Results.write("net_" + str(i) + "with accurancy: "+" test_acc = " +str(Test_result[1]) + " test_loss = " + str(Test_result[2]) +" train_acc = " + str(Train_result[1]) + " train_loss = " + str(Train_result[2]))
        Results.close()
        
        
    best_n = max(best_net) 
        
    Results = open("CNN_results.txt", "a") 
    Results.write("Final Result: " +" test_acc = "+ str(best_n[1]) + " test_loss = "+ str(best_n[2]) + "\n")
    Results.close()
    
    
    
    
    
    
    
    
    
    
    
    #-------------------------------------------------------------------------   






main()

