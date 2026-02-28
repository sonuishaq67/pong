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
    def __init__(self, env, n_envs=1, hidden_layer=512, learning_rate=0.00025, gamma=0.99, buffer_size=500000):
        self.env = env
        self.n_envs = n_envs
        self.gamma = gamma
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # Support both single env and vectorized env
        if hasattr(env, 'single_observation_space'):
            obs_shape = env.single_observation_space.shape
            n_actions = env.single_action_space.n
        else:
            obs_shape = env.observation_space.shape
            n_actions = env.action_space.n

        print('Loaded model on', self.device)

        self.memory = ReplayBuffer(max_size=buffer_size, input_shape=obs_shape, device=self.device)
        self.model = Model(action_dim=n_actions, hidden_dim=hidden_layer, observation_shape=obs_shape).to(self.device)
        self.target_model = Model(action_dim=n_actions, hidden_dim=hidden_layer, observation_shape=obs_shape).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        # Compile models for faster execution (PyTorch 2.0+)
        self.model = torch.compile(self.model, mode='reduce-overhead')
        self.target_model = torch.compile(self.target_model, mode='reduce-overhead')

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Mixed precision scaler for faster training
        self.scaler = torch.amp.GradScaler('cuda')

    def process_observation(self, obs):
        return torch.tensor(np.asarray(obs), dtype=torch.uint8, device=self.device)

    def train(self, total_episodes, summary_writer_suffix, batch_size, epsilon, epsilon_decay_steps, min_epsilon):
        summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{summary_writer_suffix}'
        writer = SummaryWriter(summary_writer_name)
        if not os.path.exists('models'):
            os.makedirs('models')

        obs_np, _ = self.env.reset()
        obs = self.process_observation(obs_np)  # (N_ENVS, 4, 84, 84) on GPU

        episode_rewards = np.zeros(self.n_envs)
        episode_start_times = [time.time()] * self.n_envs
        episodes_done = 0
        last_save = 0
        total_steps = 0

        while episodes_done < total_episodes:
            # Epsilon-greedy action selection for all envs simultaneously
            if random.random() < epsilon:
                actions = self.env.action_space.sample()  # (N_ENVS,) numpy
            else:
                with torch.no_grad():
                    q_vals = self.model(obs.float())  # (N_ENVS, n_actions)
                    actions = torch.argmax(q_vals, dim=1).cpu().numpy()

            next_obs_np, rewards, terminated, truncated, infos = self.env.step(actions)
            dones = terminated | truncated
            next_obs = self.process_observation(next_obs_np)

            # Store all N_ENVS transitions in a single vectorized write
            self.memory.store_batch(obs, actions, rewards, next_obs, dones)

            episode_rewards += rewards

            # Log each completed episode (multiple can finish in one step)
            for i in range(self.n_envs):
                if dones[i]:
                    ep_time = time.time() - episode_start_times[i]
                    writer.add_scalar('Score', episode_rewards[i], episodes_done)
                    writer.add_scalar('Epsilon', epsilon, episodes_done)
                    print(f"Episode {episodes_done} | Score: {episode_rewards[i]:.0f} | Epsilon: {epsilon:.3f} | Time: {ep_time:.1f}s")
                    episode_rewards[i] = 0
                    episode_start_times[i] = time.time()
                    episodes_done += 1

            obs = next_obs
            total_steps += 1

            # Decay epsilon by N_ENVS transitions per env-step to keep same total-transition schedule
            epsilon = max(min_epsilon, epsilon - (1.0 - min_epsilon) * self.n_envs / epsilon_decay_steps)

            if self.memory.can_sample(batch_size):
                observations, actions_b, rewards_b, next_observations, dones_b = self.memory.sample_buffer(batch_size)

                dones_b = dones_b.unsqueeze(1).float()
                actions_b = actions_b.unsqueeze(1).long()
                rewards_b = rewards_b.unsqueeze(1)

                # Mixed precision training
                with torch.amp.autocast('cuda'):
                    q_values = self.model(observations)
                    qsa_batch = q_values.gather(1, actions_b)

                    # Double DQN: online network selects actions, target network evaluates
                    with torch.no_grad():
                        next_actions = torch.argmax(self.model(next_observations), dim=1, keepdim=True)
                        next_q_values = self.target_model(next_observations).gather(1, next_actions)
                        target_b = rewards_b + (1 - dones_b) * self.gamma * next_q_values

                    loss = F.mse_loss(qsa_batch, target_b)

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if total_steps % 100 == 0:
                    writer.add_scalar("Loss/model", loss.item(), total_steps)
                    soft_update(self.target_model, self.model)

            if episodes_done >= last_save + 1000:
                self.model._orig_mod.save_model()
                last_save = episodes_done

        writer.close()

    def test(self):
        self.model._orig_mod.load_model()
        obs, info = self.env.reset()
        obs = self.process_observation(obs)

        done = False
        ep_reward = 0

        while not done:
            if random.random() < 0.05:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = self.model(obs.unsqueeze(0).float())[0]
                    action = torch.argmax(q_values, dim=-1).item()

            next_obs, reward, done, truncated, info = self.env.step(action)
            obs = self.process_observation(next_obs)
            ep_reward += reward

        print(f"Test episode finished with reward: {ep_reward}")
