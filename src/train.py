from agent import Agent
import gymnasium as gym
from gymnasium import spaces
import ale_py
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation, TransformObservation
import numpy as np
import torch

# Enable cudnn optimizations
torch.backends.cudnn.benchmark = True

episodes = 10000
max_ep_steps = 10000

# Optimized for 80GB VRAM, 32 cores, 64GB RAM
hidden_layer = 512          # Increased model capacity
learning_rate = 0.00025     # Standard DQN learning rate
step_repeat = 4
gamma = 0.99
batch_size = 512            # Much larger batch for 80GB VRAM
epsilon = 1.0
min_epsilon = 0.1
epsilon_decay_steps = 500000  # Slower decay for better exploration over 10k episodes
buffer_size = 1000000         # Larger replay buffer

# Crop function to remove score area (top 34 pixels of 210)
def crop_frame(obs):
    # Original Pong frame is 210x160, crop top 34 pixels (score area)
    # After grayscale: shape is (210, 160), crop to (176, 160)
    return obs[34:, :]

env = gym.make("ALE/Pong-v5", render_mode="rgb_array", frameskip=1)

# First convert to grayscale, then crop, then resize
env = GrayscaleObservation(env, keep_dim=False)
# Define new observation space after crop: (210-34, 160) = (176, 160)
cropped_obs_space = spaces.Box(low=0, high=255, shape=(176, 160), dtype=np.uint8)
env = TransformObservation(env, crop_frame, cropped_obs_space)
env = ResizeObservation(env, (84, 84))  # Standard DQN size
env = FrameStackObservation(env, 4)

agent = Agent(
    env,
    hidden_layer=hidden_layer,
    learning_rate=learning_rate,
    step_repeat=step_repeat,
    gamma=gamma,
    buffer_size=buffer_size
)

summary_writer_suffix = f'dqn_lr={learning_rate}_hl={hidden_layer}_bs={batch_size}'

agent.train(
    episodes=episodes,
    max_episode_steps=max_ep_steps,
    summary_writer_suffix=summary_writer_suffix,
    batch_size=batch_size,
    epsilon=epsilon,
    epsilon_decay_steps=epsilon_decay_steps,
    min_epsilon=min_epsilon
)
