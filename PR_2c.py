#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 11:08:01 2020

@author: anikajohn
"""

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])



trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)

valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)


dataiter = iter(trainloader)
images, labels = dataiter.next()

print(images.shape)
print(labels.shape)

plt.imshow(images[0].numpy().squeeze(), cmap='gray_r');

figure = plt.figure()
num_of_images = 60
for index in range(1, num_of_images + 1):
    plt.subplot(6, 10, index)
    plt.axis('off')
    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')


input_size = 784
hidden_sizes = [128, 64]
output_size = 10

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))
print(model)



criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

logps = model(images) #log probabilities
loss = criterion(logps, labels) #calculate the NLL loss


print('Before backward pass: \n', model[0].weight.grad)
loss.backward()
print('After backward pass: \n', model[0].weight.grad)

# =============================================================================
# TRAINING PROCESS
# =============================================================================
c= 0
train_losses = []
train_accuracy = []
validation_accuracy = []

for ep in range (0,31):

    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    time0 = time()
    epochs = ep
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
        
            # Training pass
            optimizer.zero_grad()
            
            output = model(images)
            loss = criterion(output, labels)
            
            #This is where the model learns by backpropagating
            loss.backward()
            
            #And optimizes its weights here
            optimizer.step()
            
            running_loss += loss.item()
        #else:
            #print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
    loss = running_loss/len(trainloader)
    #print(loss)
    train_losses.append(loss)
    
    #print(losses)
    #print("\nTraining Time (in minutes) =",(time()-time0)/60)

#print(losses)

# =============================================================================
# 
# =============================================================================
#def view_classify(img, ps):
#    ''' Function for viewing an image and it's predicted classes.
#    '''
#    ps = ps.data.numpy().squeeze()
#
#    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
#    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
#    ax1.axis('off')
#    ax2.barh(np.arange(10), ps)
#    ax2.set_aspect(0.1)
#    ax2.set_yticks(np.arange(10))
#    ax2.set_yticklabels(np.arange(10))
#    ax2.set_title('Class Probability')
#    ax2.set_xlim(0, 1.1)
#    plt.tight_layout()
#
#    
#images, labels = next(iter(valloader))
#
#img = images[3].view(1, 784) #image of which we want to test probability from validation set 
#with torch.no_grad():
#    logps = model(img)
#
#ps = torch.exp(logps)
#probab = list(ps.numpy()[0])
#print("Predicted Digit =", probab.index(max(probab)))
#view_classify(img.view(1, 28, 28), ps)  

#ACCURACY ON TRAINING SET  

    correct_count, all_count = 0, 0
    for images,labels in trainloader: #look for accuracy in validation set 
      for i in range(len(labels)):
        img = images[i].view(1, 784)
        with torch.no_grad():
            logps = model(img)
    
        
        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        if(true_label == pred_label):
          correct_count += 1
        all_count += 1
    
    #print("Number Of Images Tested =", all_count)
    #print("\nModel Accuracy =", (correct_count/all_count))

    train_acc = correct_count/all_count
    train_accuracy.append(train_acc)
        


    correct_count, all_count = 0, 0
    for images,labels in valloader: #look for accuracy in validation set 
      for i in range(len(labels)):
        img = images[i].view(1, 784)
        with torch.no_grad():
            logps = model(img)
    
        
        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        if(true_label == pred_label):
          correct_count += 1
        all_count += 1
    
    #print("Number Of Images Tested =", all_count)
    #print("\nModel Accuracy =", (correct_count/all_count))

    val_acc = correct_count/all_count
    validation_accuracy.append(val_acc)
    c = c+1
    print(c)



print(validation_accuracy)
print(train_accuracy)

epo = np.arange(1,32)
print(epo)
        
plt.plot(epo, train_accuracy, label='accuracy on training set')
plt.plot(epo, validation_accuracy, label='accuracy on validation set')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc=0)
plt.savefig("Accuracy_comp.png")
plt.show()


torch.save(model, './my_mnist_model.pt')


optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # Training pass
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        
        #This is where the model learns by backpropagating
        loss.backward()
        
        #And optimizes its weights here
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
#loss = running_loss/len(trainloader)
#print(loss)
#train_losses.append(loss)

#print(losses)
print("\nTraining Time (in minutes) =",(time()-time0)/60)

