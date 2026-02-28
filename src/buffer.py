import numpy as np
import torch


class ReplayBuffer():

    def __init__(self, max_size, input_shape, device='cpu'):
        self.mem_size = max_size
        self.mem_ctr = 0
        self.device = device

        self.state_memory = torch.zeros((self.mem_size, *input_shape), dtype=torch.uint8, device=self.device)
        self.next_state_memory = torch.zeros((self.mem_size, *input_shape), dtype=torch.uint8, device=self.device)
        self.action_memory = torch.zeros(self.mem_size, dtype=torch.long, device=self.device)
        self.reward_memory = torch.zeros(self.mem_size, dtype=torch.float32, device=self.device)
        self.terminal_memory = torch.zeros(self.mem_size, dtype=torch.bool, device=self.device)

    def can_sample(self, batch_size):
        return self.mem_ctr > batch_size * 5

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_ctr % self.mem_size
        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_ctr += 1

    def store_batch(self, states, actions, rewards, next_states, dones):
        """Vectorized store for N transitions from parallel envs in a single write."""
        n = len(actions)
        indices = (torch.arange(n, device=self.device) + self.mem_ctr) % self.mem_size
        self.state_memory[indices] = states
        self.next_state_memory[indices] = next_states
        self.action_memory[indices] = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        self.reward_memory[indices] = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        self.terminal_memory[indices] = torch.as_tensor(dones, dtype=torch.bool, device=self.device)
        self.mem_ctr += n

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_ctr, self.mem_size)
        batch = torch.randint(0, max_mem, (batch_size,), device=self.device)

        states = self.state_memory[batch].float()
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch].float()
        dones = self.terminal_memory[batch]

        return states, actions, rewards, next_states, dones
