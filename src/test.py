from agent import Agent
import gymnasium as gym
from gymnasium import spaces
import ale_py
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation, TransformObservation
import numpy as np

episodes = 10000
max_ep_steps = 10000

# Must match training parameters
hidden_layer = 512
learning_rate = 0.00025
step_repeat = 4
gamma = 0.99
batch_size = 512
epsilon = 1.0
min_epsilon = 0.1
epsilon_decay_steps = 100000

# Crop function to remove score area (top 34 pixels of 210)
def crop_frame(obs):
    return obs[34:, :]

env = gym.make("ALE/Pong-v5", render_mode="human", frameskip=1)

# Must match training preprocessing
env = GrayscaleObservation(env, keep_dim=False)
cropped_obs_space = spaces.Box(low=0, high=255, shape=(176, 160), dtype=np.uint8)
env = TransformObservation(env, crop_frame, cropped_obs_space)
env = ResizeObservation(env, (84, 84))
env = FrameStackObservation(env, 4)

agent = Agent(env, hidden_layer=hidden_layer, learning_rate=learning_rate, step_repeat=step_repeat, gamma=gamma)

agent.test()
