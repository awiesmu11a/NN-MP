from typing import Tuple
import torch as th
from torch import nn
from torch.distributions import Beta

from stable_baselines3.common.distributions import Distribution

def sum_independent_dims(tensor: th.Tensor) -> th.Tensor:
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.

    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor

class BetaDistributionAction(Distribution):

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.alpha = None
        self.beta = None
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
    
    def proba_distribution_net(self, latent_dim: int) -> Tuple[nn.Module, nn.Module]:
        self.alpha = nn.Linear(latent_dim, self.action_dim)
        self.beta = nn.Linear(latent_dim, self.action_dim)
        self.relu = nn.Softplus()

        alpha_net = [self.alpha, self.relu]
        beta_net = [self.beta, self.relu]

        return nn.Sequential(*alpha_net).to(self.device), nn.Sequential(*beta_net).to(self.device)

    def proba_distribution(self, alpha, beta):
        self.distribution = Beta(alpha, beta)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)
        pass

    def entropy(self) -> th.Tensor:
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> th.Tensor:
        return self.distribution.rsample()
    
    def mean(self) -> th.Tensor:
        return self.distribution.mean

    def mode(self) -> th.Tensor:
        return self.distribution.mean

    def actions_from_params(self, alpha, beta, deterministic: bool = False) -> th.Tensor:
        self.proba_distribution(alpha, beta)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, alpha, beta) -> Tuple[th.Tensor]:
        actions = self.actions_from_params(alpha, beta)
        log_prob = self.log_prob(actions)
        return actions, log_prob
