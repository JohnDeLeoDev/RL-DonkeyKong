import gym.spaces
from gym_dk.NESenv import SkipFrame, ResizeEnv
import torch as th
from torch import nn
from pathlib import Path
import gym
import os
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from nes_py.wrappers import JoypadSpace
from gym_dk.actions import COMPLEX_MOVEMENT
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym_dk.dk_env import DonkeyKongEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.dqn.dqn import DQN
from gym.wrappers import GrayScaleObservation  # type: ignore


def train():
    save_dir = Path("./model/DQN")
    save_dir_str = "./model/DQN"

    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    class MarioNet(BaseFeaturesExtractor):

        def __init__(self, observation_space: gym.spaces.Box, features_dim):
            super(MarioNet, self).__init__(observation_space, features_dim)
            n_input_channels = observation_space.shape[0]  # type: ignore
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
                n_flatten = self.cnn(
                    th.as_tensor(observation_space.sample()[None]).float()
                ).shape[1]

            self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

        def forward(self, observations: th.Tensor) -> th.Tensor:
            return self.linear(self.cnn(observations))

    policy_kwargs = dict(
        features_extractor_class=MarioNet,
        features_extractor_kwargs=dict(features_dim=512),
    )

    env = DonkeyKongEnv()
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    # env = CustomRewardAndDoneEnv(env)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeEnv(env, size=84)
    env = Monitor(env, save_dir_str)
    env = DummyVecEnv([lambda: env])  # type: ignore
    env = VecFrameStack(env, 4, channels_order="last")
    env.reset()

    model = DQN(
        "CnnPolicy",
        env,
        verbose=2,
        device="mps",
        tensorboard_log="./tensorboard_log/DQN",
        policy_kwargs=policy_kwargs,
        exploration_initial_eps=0,
        exploration_final_eps=0,
    )

    # render
    obs = env.reset()
    while True:
        if os.path.exists(save_dir_str + "/DQN.zip"):
            model = DQN.load(save_dir_str + "/DQN.zip")
            done = False
            while not done:
                action, _ = model.predict(obs)  # type: ignore
                obs, reward, done, info = env.step(action)
                env.render()
        else:
            print("Model not found")


if __name__ == "__main__":
    train()
