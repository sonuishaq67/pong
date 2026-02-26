import numpy as np
import torch 


class ReplayBuffer():
    
    def __init__(self,max_size, input_shape, device='cpu'):
        # storing in ram because we're storing a lot of data
        
        self.mem_size = max_size
        self.mem_ctr=0
        self.state_memory= np.zeros((self.mem_size,*input_shape),dtype=np.uint8)
        self.next_state_memory= np.zeros((self.mem_size,*input_shape),dtype=np.uint8)
        self.action_memory = np.zeros(self.mem_size, dtype=np.uint8)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size,dtype=bool) 
        self.device = device
        
    
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
        self.action_memory[index]=torch.tensor(action).detach().cpu()
        self.reward_memory[index]=reward
        self.terminal_memory[index]=done
        self.mem_ctr+=1
        
    def sample_buffer(self,batch_size):
        # max_mem to use if mem is not full
        max_mem=min(self.mem_ctr,self.mem_size)
        batch = np.random.choice(max_mem,batch_size)
        
        states=torch.tensor(self.state_memory[batch],dtype=torch.float32).to(self.device)
        actions= torch.tensor(self.action_memory[batch],dtype=torch.float32).to(self.device)
        rewards=torch.tensor(self.reward_memory[batch],dtype=torch.float32).to(self.device)
        next_states=torch.tensor(self.next_state_memory[batch],dtype=torch.float32).to(self.device)
        dones= torch.tensor(self.terminal_memory[batch],dtype=torch.bool).to(self.device)
        
        return states, actions, rewards, next_states,dones