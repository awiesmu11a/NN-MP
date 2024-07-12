from gym import Space
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class state_feature_extractor(BaseFeaturesExtractor):
    '''
    Feature extractor to process only the state-space of the agent.
    '''

    def __init__(self, observation_space: Space, features_dim: int = 4) -> None:
        
        super(state_feature_extractor, self).__init__(observation_space, features_dim)
        self.linear1 = nn.Linear(12, 8)
        self.linear2 = nn.Linear(8, features_dim)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, observations):

        observations = observations["vec"]
        x = self.relu(self.linear1(observations))
        x = self.relu(self.linear2(x))
        return x