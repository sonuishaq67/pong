import numpy as np
import torch 


class ReplayBuffer():
    
    def __init__(self,max_size, input_shape, device='cpu'):
        # storing on device to reduce RAM usage and speed up training
        
        self.mem_size = max_size
        self.mem_ctr=0
        self.device = device
        
        self.state_memory = torch.zeros((self.mem_size, *input_shape), dtype=torch.uint8, device=self.device)
        self.next_state_memory = torch.zeros((self.mem_size, *input_shape), dtype=torch.uint8, device=self.device)
        self.action_memory = torch.zeros(self.mem_size, dtype=torch.long, device=self.device)
        self.reward_memory = torch.zeros(self.mem_size, dtype=torch.float32, device=self.device)
        self.terminal_memory = torch.zeros(self.mem_size, dtype=torch.bool, device=self.device)
        
    
    # sample only when we have enough batches so as to not overfit the model on a small sample
    
    def can_sample(self,batch_size) :
        if self.mem_ctr> (batch_size*5):
            return True
        return False
    
    def store_transition(self, state, action, reward, next_state, done):
        # overwrite oldest buffer when mem is full
        index = self.mem_ctr % self.mem_size
        self.state_memory[index]= state
        self.next_state_memory[index]=next_state
        self.action_memory[index]=action
        self.reward_memory[index]=reward
        self.terminal_memory[index]=done
        self.mem_ctr+=1
        
    def sample_buffer(self,batch_size):
        # max_mem to use if mem is not full
        max_mem=min(self.mem_ctr,self.mem_size)
        batch = torch.randint(0, max_mem, (batch_size,), device=self.device)
        
        states = self.state_memory[batch].float()
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch].float()
        dones = self.terminal_memory[batch]
        
        return states, actions, rewards, next_states, dones