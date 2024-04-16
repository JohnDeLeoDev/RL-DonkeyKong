from turtle import done
import gym.spaces
from NESenv import CustomRewardAndDoneEnv, SkipFrame, ResizeEnv, display_all_frame
from typing import Any, Callable, Dict, Optional, Type, Union
from stable_baselines3.ppo.ppo import PPO
import torch as th
from torch import nn
from pathlib import Path
import datetime
from pytz import timezone
import pandas as pd
import gym
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
import gym_dk
from nes_py.wrappers import JoypadSpace
from gym_dk.actions import SIMPLE_MOVEMENT
from gym_dk.actions import COMPLEX_MOVEMENT
from gym_dk.actions import RIGHT_ONLY
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym_dk.dk_env import DonkeyKongEnv
from stable_baselines3.common.monitor import Monitor
from gym_dk.discord import send_message

def train():
    # Model Param
    CHECK_FREQ_NUMB = 1000
    TOTAL_TIMESTEP_NUMB = 5000000
    LEARNING_RATE = 0.0001
    GAE = 1.0
    ENT_COEF = 0.01
    N_STEPS = 512
    GAMMA = 0.9
    BATCH_SIZE = 64
    N_EPOCHS = 10

    # Test Param
    EPISODE_NUMBERS = 20
    MAX_TIMESTEP_TEST = 1000


    save_dir = Path('./model')
    save_dir_str = "./model"

    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    class MarioNet(BaseFeaturesExtractor):

        def __init__(self, observation_space: gym.spaces.Box, features_dim):
            super(MarioNet, self).__init__(observation_space, features_dim)
            n_input_channels = observation_space.shape[0] # type: ignore
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )

            # Compute shape by doing one forward pass
            with th.no_grad():
                n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

            self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

        def forward(self, observations: th.Tensor) -> th.Tensor:
            return self.linear(self.cnn(observations))

    policy_kwargs = dict(
        features_extractor_class=MarioNet,
        features_extractor_kwargs=dict(features_dim=512),
    )

    class TrainAndLoggingCallback(BaseCallback):
        def __init__(self, check_freq, save_path, verbose=1):
            super(TrainAndLoggingCallback, self).__init__(verbose)
            self.check_freq = check_freq
            self.save_path = save_path

        def _init_callback(self):
            if self.save_path is not None:
                os.makedirs(self.save_path, exist_ok=True)

        def _on_step(self):            
            if self.n_calls % self.check_freq == 0:
                model_path = (save_dir / 'PPO.zip')
                self.model.save(model_path) # type: ignore
                    
            return True
    
    callback = TrainAndLoggingCallback(check_freq=CHECK_FREQ_NUMB, save_path=save_dir)

    STAGE_NAME = 'DonkeyKong-v0' 
    MOVEMENT = COMPLEX_MOVEMENT

    env = DonkeyKongEnv()
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    #env = CustomRewardAndDoneEnv(env)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeEnv(env, size=84)
    env = Monitor(env, save_dir_str)
    env = DummyVecEnv([lambda: env]) # type: ignore
    env = VecFrameStack(env, 4, channels_order='last')
    env.reset()
   
    model = PPO('CnnPolicy', env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log=save_dir, learning_rate=LEARNING_RATE, n_steps=N_STEPS, # type: ignore
              batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, gamma=GAMMA, gae_lambda=GAE, ent_coef=ENT_COEF, device='mps')
    
    if os.path.exists('model/PPO.zip'):
        model = PPO.load('model/PPO', env=env)
        print('Model is loaded')

    send_message('Training is started.')
    # print the results of below
    model.learn(total_timesteps=TOTAL_TIMESTEP_NUMB, callback=callback)



    send_message('Training is done!')

if __name__ == "__main__":
    train()
    # parallelTestModule.test()