import sys
import argparse
import json
import importlib
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from  environment import Env # Look into this, the names of environment and what kind of environment

from feature_extractor import stateOnly
from policy import pi

def train_network(args):

    with open("config/config_main.json", "r") as json_file:
        
        config = json.load(json_file)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stats_path = config["utils"]["env"]
    tb_path = config["utils"]["tensorboard"]
    models_path = config["utils"]["models"]

    def make_env(obs):
        return Env(obs=obs, render=False, eval=False)
    
    num_envs = config["implement"]["num_envs"]
    env = make_env(config["obs"])
    env = Monitor(env, stats_path)
    env.reset()

    env = SubprocVecEnv([make_env for _ in range(num_envs)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True)

    # Do something such that this is done on the basis of the distribution.
    policy = dict(
        net_arch = dict(pi = config["policy"]["actor"], vf = config["policy"]["critic"]),
        activation_fn = torch.nn.ReLU,
        features_extractor_class = stateOnly, # Change name
        distribution=config["policy"]["distribution"],
        log_std_init = config["policy"]["log_std_init"],
        squash_output=False
    )

    algo = config # Make the regularisation as an argument

    model = algo(
        policy = pi,
        n_steps = config["algorithm"]["n_steps"],
        learning_rate = config["algorithm"]["learning_rate"],
        n_epochs = config["algorithm"]["n_epochs"],
        env = env,
        batch_size=config["algorithm"]["batch_size"],
        clip_range=config["algorithm"]["clip_range"],
        ent_coef=config["algorithm"]["ent_coeff"],
        vf_coef=config["algorithm"]["vf_coeff"],
        policy_kwargs=policy,
        verbose=0,
        tensorboard_log=config["utils"]["tensorboard"],
        seed=1
    )

    # Do something about training the model which is saved
    save_model_partial = None
    if save_model_partial:
        checkpoint = CheckpointCallback(save_freq=config["algorithm"]["save_freq"],
                                        save_path=config["utils"]["models"],
                                        name_prefix=args.file, verbose=0)
    
    model.learn(total_timesteps=config, callback=checkpoint)
    rewards = env.get_total_original_reward()
    # Plot the rewards in a presentable manner
    model.save(config)
    env.save(config)

def evaluate_network(args):

    with open("config/config.json", "r") as json_file:
        # Wherever config is present, fill the statement i sometime
        config = json.load(json_file)

    env = config
    env=Monitor(env)
    env=DummyVecEnv([lambda: env])
    stats_path = os.path.join(config)
    env=VecNormalize.load() # Add path
    env.training=False
    env.norm_reward=False

    # Look how the results should be presented
    # Load the model and run the episode
    # Lipschitz regularization is left, incorporating it in algorithm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--file')
    parser.add_argument('--eval', type=bool, default=False, help='Specify if running in evaluation mode.')
    args = parser.parse_args()
    if not args.eval:
        train_network(args)
    else:
        evaluate_network(args)