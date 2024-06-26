import gym.spaces
from gym_dk.NESenv import SkipFrame, ResizeEnv
from stable_baselines3.ppo.ppo import PPO
import torch as th
from torch import nn
from pathlib import Path
import gym
import os
from gym.wrappers import GrayScaleObservation  # type: ignore
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from nes_py.wrappers import JoypadSpace
from gym_dk.actions import COMPLEX_MOVEMENT
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym_dk.dk_env import DonkeyKongEnv
from stable_baselines3.common.monitor import Monitor


def train():
    """
    Train the PPO model for Donkey Kong game.

    This function initializes the necessary parameters, creates the MarioNet class for feature extraction,
    sets up the training environment, and trains the PPO model using the specified parameters.

    Returns:
        None
    """
    # Model Param
    CHECK_FREQ_NUMB = 1000
    TOTAL_TIMESTEP_NUMB = 1000000
    LEARNING_RATE = 0.0001
    GAE = 1.0
    ENT_COEF = 0.01
    N_STEPS = 512
    GAMMA = 0.9
    BATCH_SIZE = 512
    N_EPOCHS = 10

    save_dir = Path("./model/PPO")
    save_dir_str = "./model/PPO"

    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    class MarioNet(BaseFeaturesExtractor):
        """
        Convolutional neural network for feature extraction in the PPO algorithm.

        Args:
            observation_space (gym.spaces.Box): The observation space of the environment.
            features_dim (int): The dimension of the extracted features.

        Attributes:
            cnn (nn.Sequential): The convolutional layers of the network.
            linear (nn.Sequential): The linear layers of the network.

        """

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
            """
            Forward pass of the network.

            Args:
                observations (th.Tensor): The input observations.

            Returns:
                th.Tensor: The extracted features.

            """
            return self.linear(self.cnn(observations))

    policy_kwargs = dict(
        features_extractor_class=MarioNet,
        features_extractor_kwargs=dict(features_dim=512),
    )

    class TrainAndLoggingCallback(BaseCallback):
        """
        Callback for training and logging.

        This callback is used to save the model at regular intervals during training.

        :param check_freq: The frequency at which to save the model (in number of calls to `_on_step`).
        :param save_path: The path to save the model.
        :param verbose: Verbosity level (0: no output, 1: info).
        """

        def __init__(self, check_freq, save_path, verbose=1):
            super(TrainAndLoggingCallback, self).__init__(verbose)
            self.check_freq = check_freq
            self.save_path = save_path

        def _init_callback(self):
            """
            Initialize the callback.
            Create the save path if it doesn't exist.
            """
            if self.save_path is not None:
                os.makedirs(self.save_path, exist_ok=True)

        def _on_step(self):
            """
            Perform an action at each training step.
            Save the model if the number of calls to `_on_step` is a multiple of `check_freq`.

            :return: True to continue training, False to stop training.
            """
            if self.n_calls % self.check_freq == 0:
                model_path = save_dir / "PPO.zip"
                self.model.save(model_path)  # type: ignore

            return True

    callback = TrainAndLoggingCallback(check_freq=CHECK_FREQ_NUMB, save_path=save_dir)

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

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        tensorboard_log=save_dir,  # type: ignore
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,  # type: ignore
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE,
        ent_coef=ENT_COEF,
        device="mps",
    )

    while True:
        # check that file is a zip file
        if os.path.exists("model/PPO/PPO.zip"):
            model = PPO.load("model/PPO/PPO.zip")
            state = env.reset()
            done = False
            while not done:
                action, _ = model.predict(state)  # type: ignore
                state, reward, done, info = env.step(action)
                env.render()


if __name__ == "__main__":
    train()
