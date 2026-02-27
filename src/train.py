from agent import Agent
import gymnasium as gym
import ale_py
from gymnasium.wrappers import GrayscaleObservation,ResizeObservation,FrameStackObservation

episodes=10000
max_ep_steps=10000

hidden_layer=128
learning_rate=0.0001
step_repeat=4
gamma=0.99
batch_size=64
epsilon=1
min_epsilon=0.1
epsilon_decay_steps=100000


env=gym.make("ALE/Pong-v5",render_mode="rgb_array",frameskip=1)

env= ResizeObservation(env,(64,64))
env= GrayscaleObservation(env, keep_dim=False)
env= FrameStackObservation(env, 4)

agent = Agent(env,hidden_layer=hidden_layer,learning_rate=learning_rate,step_repeat=step_repeat,gamma=gamma)

summary_writer_suffix=f'dqn_lr={learning_rate}_hl={hidden_layer}_bs={batch_size}'

agent.train(episodes=episodes,max_episode_steps=max_ep_steps,summary_writer_suffix=summary_writer_suffix, batch_size=batch_size,epsilon=epsilon,epsilon_decay_steps=epsilon_decay_steps,min_epsilon=min_epsilon)
