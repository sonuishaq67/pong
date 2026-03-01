from agent import Agent
import gymnasium as gym
from gymnasium import spaces
import ale_py
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation, TransformObservation
import numpy as np

# Must match training parameters
hidden_layer = 512
learning_rate = 0.00025
gamma = 0.99


def crop_frame(obs):
    return obs[34:, :]


env = gym.make("ALE/Pong-v5", render_mode="human", frameskip=4)

env = GrayscaleObservation(env, keep_dim=False)
cropped_obs_space = spaces.Box(low=0, high=255, shape=(176, 160), dtype=np.uint8)
env = TransformObservation(env, crop_frame, cropped_obs_space)
env = ResizeObservation(env, (84, 84))
env = FrameStackObservation(env, 4)

agent = Agent(env, n_envs=1, hidden_layer=hidden_layer, learning_rate=learning_rate, gamma=gamma,buffer_size=1)

agent.test()
