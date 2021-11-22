#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import os
import nltk

print("git")


# In[2]:


df = pd.read_csv('data/car.data',header=None, usecols=[0,1,2,4,5,6], names=['buying_price', 'maint', 
                                                                            'doors', 'lug_boot_size',
                                                                           'safety','class_value'])
data = df[['maint', 'doors', 'lug_boot_size','safety','class_value']]
target = df[['buying_price']]


# In[3]:


# categorized data features that are not in numeric value
data = pd.concat([data, pd.get_dummies(data.maint)], axis=1)
data = pd.concat([data, pd.get_dummies(data.doors)], axis=1)
data = pd.concat([data, pd.get_dummies(data.lug_boot_size)], axis=1)
data = pd.concat([data, pd.get_dummies(data.safety)], axis=1)
data = pd.concat([data, pd.get_dummies(data.class_value)], axis=1)
data = data.drop([ 'maint', 'doors', 'lug_boot_size', 'safety','class_value'], axis=1)
display(data.head())


# In[4]:


target= pd.concat([target, pd.get_dummies(target.buying_price)], axis=1)
target = target.drop(['buying_price'], axis=1) 
#target['buying_price'] = target['buying_price'].factorize()[0]
display(target.head())


# In[5]:


#prep dataloader for training 
train_dataset, valid_dataset, train_targets, valid_targets = train_test_split(data,target, test_size = 0.05,
                                                                                   random_state=1)
train_dataset_tensor = torch.Tensor(data.values)
train_target_tensor = torch.Tensor(target.values)

train_dataset_tensor = torch.Tensor(train_dataset.values)
train_target_tensor = torch.Tensor(train_targets.values)
valid_dataset_tensor = torch.Tensor(valid_dataset.values)
valid_target_tensor = torch.Tensor(valid_targets.values)
  
from torch.utils import data
train_dataset = data.TensorDataset(train_dataset_tensor, train_target_tensor)
valid_dataset = data.TensorDataset(valid_dataset_tensor, valid_target_tensor)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size = 128, num_workers =2, shuffle = False,
                                    )

val_dataloader = torch.utils.data. DataLoader(
    valid_dataset, batch_size = 128, num_workers =2, shuffle = False,
                                )  

                                  


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[13]:


#build model
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self, num_classes=4):
        super(Net, self).__init__()
        self.fc1=nn.Linear(18, 18)#18 is the number of data feature column
        self.fc2=nn.Linear(18, 64)
 
        self.fc=nn.Linear(in_features=64, out_features=num_classes)
        
    
    def forward(self, x):
        x = x.view(-1, 18)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        
        x=self.fc(x)
        x=F.softmax(x, dim=1)
        return x

net=Net()
net.to(device)
    


# In[ ]:


# train model and validate
learning_rate = 0.02
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
n_epochs = 300

running_loss = 0
net.train()

for epoch in range(n_epochs):
    for train, target in train_dataloader:
        net.zero_grad()
        train, target = train.to(device), target.to(device)
         
        output = net.forward(train)
        #print(output, target)
        loss = criterion(output, torch.max(target, 1)[1])
        loss.backward()
        optimizer.step()
   
    if epoch % 20 ==0:
        for val, target in val_dataloader:

            val, target = val.to(device), target.to(device)
            output = net.forward(val)
                #print(output, target)
            val_loss = criterion(output, torch.max(target, 1)[1])

            print('Val loss ' + str(val_loss.detach().item()))
        print('Epoch '+ str(epoch) + '.. Training loss ' + str(loss.detach().item()))


# In[9]:


#save model for later output
torch.save(net, 'model.pth')
model = torch.load( 'model.pth')
model.eval()
#Class Value: 'Good'
#Safety: 'High' 
#Maintenance: 'High' 
#Number of doors : 4 
#Lug boot size: 'Big'
    
#convert test input format into model format:
input_tensor = torch.Tensor([1,0,0,0,0,0,1,0,1,0,0,1,0,0,0,1,0,0]).to(device)
out_logit = model.forward(input_tensor)
 
# 0,1,2,3 is corresponding to high, low, med,vhigh
output_label = out_logit.data.cpu().numpy().argmax()
label_dict = {0: "high", 1: "low", 2: "med", 3: "vhigh"}
output =  label_dict[output_label]
print("buying price is", output)

