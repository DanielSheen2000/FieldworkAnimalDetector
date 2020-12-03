import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms

class Net(nn.Module):
    def __init__(self, mfcc_total):
        super(Net, self).__init__()
        self.conv_mfcc = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(mfcc_total, 10), stride=5)
        self.conv_delta = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(mfcc_total, 10), stride=5)
        self.conv_delta_delta = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(mfcc_total, 10), stride=5)

        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(3,10), stride=3)
        self.conv_2 = nn.Conv2d(in_channels=10, out_channels=5, kernel_size=(1,10), stride=1)
        self.conv_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1,10),stride=1)
        self.conv_4 = nn.Conv2d(in_channels=256,out_channels = 512, kernel_size = (1,3),stride=2)
        self.fc1 = nn.Linear(45, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, mfcc, delta, delta_delta):
        # Compressing features into 3x300 matrix
        mfcc_features = self.conv_mfcc(mfcc)
        delta_features = self.conv_delta(delta)
        delta_delta_features = self.conv_delta_delta(delta_delta)
        features = torch.cat((mfcc_features, delta_features, delta_delta_features),2)
        features = F.relu(self.conv_1(features))
        features = F.relu(self.conv_2(features))
        #features = F.relu(self.conv_3(features))
        #features = F.relu(self.conv_4(features)).squeeze()
        features = features.view(-1,85)
        features = self.fc1(features)
        features = torch.sigmoid(self.fc2(features))
        return features

def load_model(lr, seed, mfcc_total):
    torch.manual_seed(seed)
    model = Net(mfcc_total)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, loss_function, optimizer

def load_trained():
    model = Net(13)
    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    return model