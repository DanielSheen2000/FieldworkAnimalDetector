#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import unicode_literals
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import wave
import sys
import librosa
import librosa.display
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


# ### Obtaining Data

# In[2]:


mfccs = []
deltas = []
delta_deltas = []
label = []
X = []
path_hyena = 'data/processed data/hyenas/'
path_lion = 'data/processed data/lions/'
min_shape = 1000

for file in os.listdir(path_hyena):
    hyena = np.load(path_hyena + file)
    if min_shape > hyena[0].shape[1]:
        min_shape = hyena[0].shape[1]
    mfcc = np.asarray(hyena[0][:, :300])
    delta = np.asarray(hyena[1][:, :300])
    delta_delta = np.asarray(hyena[2][:, :300])
    mfccs.append(mfcc)
    deltas.append(delta)
    delta_deltas.append(delta_delta)
    X.append([mfcc, delta, delta_delta])
    label.append(0)
    
for file in os.listdir(path_lion):
    lion = np.load(path_lion + file)
    if min_shape > lion[0].shape[1]:
        min_shape = lion[0].shape[1]
    mfcc = np.asarray(lion[0][:, :300])
    delta = np.asarray(lion[1][:, :300])
    delta_delta = np.asarray(lion[2][:, :300])
    mfccs.append(mfcc)
    deltas.append(delta)
    delta_deltas.append(delta_delta)
    X.append([mfcc, delta, delta_delta])
    label.append(1)

print('Minimum length all of our tensors is: %i' % (min_shape))


# So now we have obtained the following for our input data:
# 1. **mfccs[i]**: the mel-frequency cepstrum coefficients for a single audio file indexed at *i*
# 2. **deltas[i]**: the first derivative of the mfccs 
# 3. **delta_deltas[i]**: the second derivative of the mfccs
# 
# and our associated label:
# 1. **label[i]**: the audio file indexed at *i* will be 1 if it is a lion and 0 if it is a hyena
# 
# Now splitting our data and randomizing it:

# In[3]:


split_test = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state=0)
split_val = split_1 = StratifiedShuffleSplit(n_splits = 1, test_size = 0.1, random_state=0)
X = np.asarray(X)
label = np.asarray(label)

for train_index, test_index in split_test.split(X, label):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = label[train_index], label[test_index]
        

for train_index, test_index in split_val.split(X_train, y_train):
    X_train, X_val = X_train[train_index], X_train[test_index]
    y_train, y_val = y_train[train_index], y_train[test_index]
    
# printing out number of classes in each
print('Training has %d samples of class 0 and %d samples of class 1' % (np.unique(y_train, return_counts=True)[1][0], np.unique(y_train, return_counts=True)[1][1]))
print('Validation has %d samples of class 0 and %d samples of class 1' % (np.unique(y_val, return_counts=True)[1][0], np.unique(y_val, return_counts=True)[1][1]))
print('Testing has %d samples of class 0 and %d samples of class 1' % (np.unique(y_test, return_counts=True)[1][0], np.unique(y_test, return_counts=True)[1][1]))


# ### Network

# In[66]:


class Net(nn.Module):
    def __init__(self, mfcc_total):
        super(Net, self).__init__()
        self.conv_mfcc = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = (mfcc_total,10), stride = 5)
        self.conv_delta = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = (mfcc_total,10), stride = 5)
        self.conv_delta_delta = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = (mfcc_total,10), stride = 5)
        
        self.conv_1 = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = (3,5), stride = 2)
        self.conv_2 = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = (3,5), stride = 1)
        self.fc1 = nn.Linear(28, 10)
        self.fc2 = nn.Linear(10, 1)
        self.fc3 = nn.Sigmoid()
        
    def forward(self, mfcc, delta, delta_delta):
        # Compressing features into 3x300 matrix
        mfcc_features = self.conv_mfcc(mfcc)
        delta_features = self.conv_delta(delta)
        delta_delta_features = self.conv_delta_delta(delta_delta)
        features = torch.cat((mfcc_features, delta_features, delta_delta_features), 2)
        
        features = self.conv_1(features)
        features = features.view(-1, 28)
        features = self.fc1(features).squeeze(0)
        features = self.fc2(features)
        features = self.fc3(features)
        return features
    
def load_model(lr, seed, mfcc_total):
    torch.manual_seed(seed)
    model = Net(mfcc_total)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    return model, loss_function, optimizer


# ### Training

# In[8]:


###### SET HYPERPARAMETERS HERE ######
lr = 0.01
seed = 42
mfcc_total = 13
epochs = 1


# In[62]:


def main_test():
    torch.manual_seed(seed)
    model, loss_function, optimizer = load_model(lr, seed, mfcc_total)
    for epoch in range(epochs):
        for i in range(len(X_train)):
            # Obtaining our mfcc, delta and delta_delta from X and converting to tensor
            mfcc = torch.from_numpy(X[i][0]).unsqueeze(0).unsqueeze(0)
            delta = torch.from_numpy(X[i][1]).unsqueeze(0).unsqueeze(0)
            delta_delta = torch.from_numpy(X[i][2]).unsqueeze(0).unsqueeze(0)
            model(mfcc, delta, delta_delta)
            
            ###### FILL THIS OUT ######
    
        
main_test()


# In[ ]:





# In[ ]:





# In[ ]:




