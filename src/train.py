from agent import Agent
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import AsyncVectorEnv
import ale_py
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation, TransformObservation
import numpy as np
import torch

# Enable cudnn optimizations
torch.backends.cudnn.benchmark = True

N_ENVS = 16                     # 16 parallel envs on 16 cores, 16 left for PyTorch
total_episodes = 60000

# Optimized for 80GB VRAM, 32 cores, 64GB RAM
hidden_layer = 512
learning_rate = 0.00025
gamma = 0.99
batch_size = 512                # Reduced: halves compute per update vs 1024
epsilon = 1.0
min_epsilon = 0.1
epsilon_decay_steps = 2000000   # Scaled ~4x for 60K episodes: more exploration = faster episodes
buffer_size = 250000            # Reduced: less VRAM, faster random access (~14GB)
train_freq = 4                  # Train every 4 env-steps instead of every step


def crop_frame(obs):
    return obs[34:, :]


def make_env():
    def _init():
        env = gym.make("ALE/Pong-v5", frameskip=4)
        env = GrayscaleObservation(env, keep_dim=False)
        cropped_obs_space = spaces.Box(low=0, high=255, shape=(176, 160), dtype=np.uint8)
        env = TransformObservation(env, crop_frame, cropped_obs_space)
        env = ResizeObservation(env, (84, 84))
        env = FrameStackObservation(env, 4)
        return env
    return _init


vec_env = AsyncVectorEnv([make_env() for _ in range(N_ENVS)])

agent = Agent(
    vec_env,
    n_envs=N_ENVS,
    hidden_layer=hidden_layer,
    learning_rate=learning_rate,
    gamma=gamma,
    buffer_size=buffer_size
)

summary_writer_suffix = f'dqn_lr={learning_rate}_hl={hidden_layer}_bs={batch_size}_n_envs={N_ENVS}'

agent.train(
    total_episodes=total_episodes,
    summary_writer_suffix=summary_writer_suffix,
    batch_size=batch_size,
    epsilon=epsilon,
    epsilon_decay_steps=epsilon_decay_steps,
    min_epsilon=min_epsilon,
    train_freq=train_freq
)
