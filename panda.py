import gymnasium as gym
import panda_gym
import numpy as np
import datetime
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import TQC

base_path = 'panda-icm-seed-42/'

env = gym.make("PandaPickAndPlace-v3")

model = TQC(
    "MultiInputPolicy",
    env,
    batch_size=2048,
    buffer_size=1000000,
    gamma=0.95,
    learning_rate=0.001,
    policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(goal_selection_strategy='future', n_sampled_goal=4),
    tau=0.05,
    seed=42,
    verbose=1,
    tensorboard_log=f'{base_path}/tensorboard/',
    inum=0,
)

stringified_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
checkpoint_callback = CheckpointCallback( 
    save_freq=10_000,
    save_path=f"{base_path}/models/{stringified_time}/", 
    name_prefix="tqc_panda_pick_and_place"
)  # Callback for saving the model

# Model training: 
model.learn(
    total_timesteps=1_000_000.0,
    callback=checkpoint_callback, 
    progress_bar=True
)


#seed=3157870761,42,1