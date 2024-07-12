import torch.nn as nn
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import matplotlib.pyplot as plt
import numpy as np

class mlp_img_state(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim = 576):

        super(mlp_img_state,self).__init__(observation_space, features_dim)
                
        self.conv1 = nn.Conv2d(4, 32, (4,4), stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 32, (4,4), stride=1)
        self.fc1 = nn.Linear(32 * 13 * 13, 1024)
        self.fc2 = nn.Linear(1024, 512) 

        self.linear_state = nn.Linear(12, 64)
        self.linear_state2 = nn.Linear(64, 64)
    
    def forward(self, observations):
        img = observations["img"]
        vec = observations["vec"]
        x = self.relu(self.conv1(img))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)

        x = torch.flatten(x, 1, -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))


        state_features = self.relu(self.linear_state2(self.relu(self.linear_state(vec))))
        features = torch.cat([x, state_features], dim=1)
        return features