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
from local_model import *
from torch.utils import data

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


split_test = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
split_val = split_1 = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
X = np.asarray(X)
label = np.asarray(label)

for train_index, test_index in split_test.split(X, label):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = label[train_index], label[test_index]

for train_index, test_index in split_val.split(X_train, y_train):
    X_train, X_val = X_train[train_index], X_train[test_index]
    y_train, y_val = y_train[train_index], y_train[test_index]

# printing out number of classes in each
print('Training has %d samples of class 0 and %d samples of class 1' % (
np.unique(y_train, return_counts=True)[1][0], np.unique(y_train, return_counts=True)[1][1]))
print('Validation has %d samples of class 0 and %d samples of class 1' % (
np.unique(y_val, return_counts=True)[1][0], np.unique(y_val, return_counts=True)[1][1]))
print('Testing has %d samples of class 0 and %d samples of class 1' % (
np.unique(y_test, return_counts=True)[1][0], np.unique(y_test, return_counts=True)[1][1]))

class testingDataset(data.Dataset):
    def __init__(self, data, labels):
        self.labels = labels
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        sample_label = self.labels[index]
        return sample, sample_label

seed = 42
batch_size = 20
def main_test():
    torch.manual_seed(seed)
    model, loss_function, optimizer = load_model(lr, seed, mfcc_total)
    train_dataset = testingDataset(X_train,y_train)
    train_loader = data.DataLoader(train_dataset,batch_size = batch_size, shuffle=True)
    valid_dataset = testingDataset(X_val,y_val)
    valid_loader = data.DataLoader(valid_dataset,batch_size = batch_size)
    tloss = []
    tacc = []
    vloss = []
    vacc = []
    for epoch in range(epochs):
        print(epoch)
        taccumloss = 0
        tcorrect = 0
        ttotal = 0
        tnumberofBatches = 0
        vnumberofBatches = 0
        vaccumloss = 0
        vcorrect = 0
        vtotal = 0
        for i, batch in enumerate(train_loader):
            tmp_prediction = []
            input, label = batch
            optimizer.zero_grad()
            # Obtaining our mfcc, delta and delta_delta from X and converting to tensor
            for j in range(len(input)):
                mfcc = input[j][0].unsqueeze(0).unsqueeze(0)
                delta = input[j][1].unsqueeze(0).unsqueeze(0)
                delta_delta = input[j][2].unsqueeze(0).unsqueeze(0)
                prediction = model(mfcc,delta,delta_delta)
                tmp_prediction.append(prediction)
            print("Prediction:",tmp_prediction)
            print("Label:",label)
            tmp_prediction = torch.tensor(tmp_prediction,requires_grad = True)
            loss = loss_function(input = tmp_prediction,target = label.float())
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            for batch in train_loader:
                tmp_prediction = []
                input, label = batch
                optimizer.zero_grad()
                # Obtaining our mfcc, delta and delta_delta from X and converting to tensor
                for j in range(len(input)):
                    mfcc = input[j][0].unsqueeze(0).unsqueeze(0)
                    delta = input[j][1].unsqueeze(0).unsqueeze(0)
                    delta_delta = input[j][2].unsqueeze(0).unsqueeze(0)
                    prediction = model(mfcc, delta, delta_delta)
                    tmp_prediction.append(prediction)
                tmp_prediction = torch.tensor(tmp_prediction, requires_grad=True)
                loss = loss_function(input=tmp_prediction, target=label.float())
                taccumloss += loss.item()
                for i in range(len(tmp_prediction)):
                    if tmp_prediction[i] > 0.5 and label[i] == 1:
                        tcorrect += 1
                    elif tmp_prediction[i] <= 0.5 and label[i] <= 1:
                        tcorrect += 1
                ttotal += label.float().size(0)
                tnumberofBatches += 1
        with torch.no_grad():
            for batch in valid_loader:
                tmp_prediction = []
                input, label = batch
                optimizer.zero_grad()
                # Obtaining our mfcc, delta and delta_delta from X and converting to tensor
                for j in range(len(input)):
                    mfcc = input[j][0].unsqueeze(0).unsqueeze(0)
                    delta = input[j][1].unsqueeze(0).unsqueeze(0)
                    delta_delta = input[j][2].unsqueeze(0).unsqueeze(0)
                    prediction = model(mfcc, delta, delta_delta)
                    tmp_prediction.append(prediction)
                tmp_prediction = torch.tensor(tmp_prediction, requires_grad=True)
                loss = loss_function(input=tmp_prediction, target=label.float())
                vaccumloss += loss.item()
                for i in range(len(tmp_prediction)):
                    if tmp_prediction[i] > 0.5 and label[i] == 1:
                        vcorrect += 1
                    elif tmp_prediction[i] <= 0.5 and label[i] <= 1:
                        vcorrect += 1
                vtotal += label.float().size(0)
                vnumberofBatches += 1
        tacc.append(tcorrect/ttotal)
        tloss.append(taccumloss/tnumberofBatches)
        vacc.append(vcorrect / vtotal)
        vloss.append(vaccumloss / vnumberofBatches)
            ###### FILL THIS OUT ######
    return tloss, tacc, vloss, vacc

tloss, tacc, vloss, vacc = main_test()
epoch = range(20)
plt.plot(epoch, tloss, label='Training Loss')
print(tloss)
print(vloss)
print(tacc)
print(vacc)