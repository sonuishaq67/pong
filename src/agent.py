from buffer import ReplayBuffer
from model import Model, soft_update
import torch
import torch.optim as optim
import torch.nn.functional as F
import datetime
import time
from torch.utils.tensorboard import SummaryWriter
import random, os
import numpy as np

class Agent():
    def __init__(self, env, hidden_layer, learning_rate, step_repeat, gamma, buffer_size=1000000):
        self.env = env
        self.step_repeat = step_repeat
        self.gamma = gamma
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        obs, info = self.env.reset()
        obs = self.process_observation(obs)

        print('Loaded model on', self.device)

        self.memory = ReplayBuffer(max_size=buffer_size, input_shape=obs.shape, device=self.device)
        self.model = Model(action_dim=env.action_space.n, hidden_dim=hidden_layer, observation_shape=obs.shape).to(self.device)
        self.target_model = Model(action_dim=env.action_space.n, hidden_dim=hidden_layer, observation_shape=obs.shape).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        # Compile models for faster execution (PyTorch 2.0+)
        self.model = torch.compile(self.model, mode='reduce-overhead')
        self.target_model = torch.compile(self.target_model, mode='reduce-overhead')

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.learning_rate = learning_rate

        # Mixed precision scaler for faster training
        self.scaler = torch.amp.GradScaler('cuda')
        
    def process_observation(self,obs):
        obs=torch.tensor(np.array(obs),dtype=torch.uint8,device=self.device)
        return obs
    
    def train(self, episodes, max_episode_steps, summary_writer_suffix, batch_size, epsilon, epsilon_decay_steps, min_epsilon):
        summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{summary_writer_suffix}'
        writer = SummaryWriter(summary_writer_name)
        if not os.path.exists('models'):
            os.makedirs('models')
        total_steps = 0

        for episode in range(episodes):
            done = False
            episode_reward = 0
            obs, info = self.env.reset()
            obs = self.process_observation(obs)

            episode_steps = 0
            episode_start_time = time.time()

            while not done and episode_steps < max_episode_steps:
                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        q_values = self.model(obs.unsqueeze(0).float())[0]
                        action = torch.argmax(q_values, dim=-1).item()

                reward = 0

                for _ in range(self.step_repeat):
                    next_obs, reward_temp, done, truncated, info = self.env.step(action=action)
                    reward += reward_temp
                    if done:
                        break

                next_obs = self.process_observation(next_obs)

                self.memory.store_transition(obs, action, reward, next_obs, done)

                obs = next_obs

                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                # Linear epsilon decay
                epsilon = max(min_epsilon, epsilon - (1.0 - min_epsilon) / epsilon_decay_steps)

                if self.memory.can_sample(batch_size):
                    observations, actions, rewards, next_observations, dones = self.memory.sample_buffer(batch_size)

                    dones = dones.unsqueeze(1).float()
                    actions = actions.unsqueeze(1).long()
                    rewards = rewards.unsqueeze(1)

                    # Mixed precision training
                    with torch.amp.autocast('cuda'):
                        # Current Q values
                        q_values = self.model(observations)
                        qsa_batch = q_values.gather(1, actions)

                        # Double DQN: use online network to select actions, target network to evaluate
                        with torch.no_grad():
                            next_actions = torch.argmax(self.model(next_observations), dim=1, keepdim=True)
                            next_q_values = self.target_model(next_observations).gather(1, next_actions)
                            target_b = rewards + (1 - dones) * self.gamma * next_q_values

                        loss = F.mse_loss(qsa_batch, target_b)

                    # Scaled backward pass
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()

                    # Gradient clipping for stability
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    # Log loss
                    if total_steps % 100 == 0:
                        writer.add_scalar("Loss/model", loss.item(), total_steps)

                    # Soft update target network
                    if episode_steps % 4 == 0:
                        soft_update(self.target_model, self.model)

            if episode % 1000 == 0:
                self.model._orig_mod.save_model()  # Access original model through compiled wrapper

            writer.add_scalar('Score', episode_reward, episode)
            writer.add_scalar('Epsilon', epsilon, episode)

            episode_time = time.time() - episode_start_time

            print(f"Episode {episode} | Score: {episode_reward:.0f} | Epsilon: {epsilon:.3f} | Steps: {episode_steps} | Time: {episode_time:.1f}s")
            
            
    def test(self):
        self.model._orig_mod.load_model()  # Access original model through compiled wrapper
        obs, info = self.env.reset()

        done = False
        obs, info = self.env.reset()
        obs = self.process_observation(obs)
        ep_reward = 0

        while not done:
            if random.random() < 0.05:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = self.model(obs.unsqueeze(0).float())[0]
                    action = torch.argmax(q_values, dim=-1).item()

            reward = 0

            for _ in range(self.step_repeat):
                next_obs, reward_temp, done, truncated, info = self.env.step(action=action)
                reward += reward_temp
                if done:
                    break

            obs = self.process_observation(next_obs)

            ep_reward += reward

        print(f"Test episode finished with reward: {ep_reward}")