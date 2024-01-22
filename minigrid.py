import minigrid
from minigrid.wrappers import ImgObsWrapper,FlatObsWrapper,bWrapper
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO,DQN
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        print(n_input_channels)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2),padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2),padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2),padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample_image = observation_space.sample()
            sample_image = torch.as_tensor(sample_image).float()
            sample_image = sample_image.unsqueeze(0)  
            n_flatten = self.cnn(torch.as_tensor(sample_image).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
policy_kwargs = dict(
                    features_extractor_class=MinigridFeaturesExtractor,
                    features_extractor_kwargs=dict(features_dim=256),)
env = gym.make("MiniGrid-BlockedUnlockPickup-v0",render_mode="rgb_array")
env = ImgObsWrapper(env)

model = PPO("CnnPolicy",env,tensorboard_log='mini-ppo-seed-1',policy_kwargs=policy_kwargs,verbose=1,seed=1)
model.learn(1e6)
