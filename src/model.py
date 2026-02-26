import torch 
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, action_dim, hidden_dim=256, observation_shape=None):
        super(Model,self).__init__()
        
        self.conv1=nn.Conv2d(in_channels=1,out_channels=8,kernel_size=4, stride=2)
        self.conv2=nn.Conv2d(in_channels=8,out_channels=16,kernel_size=4, stride=2)
        self.conv3=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3, stride=2)

        conv_outp_size=self.calc_conv_output(observation_shape)
        print("conv output size",conv_outp_size)


        self.fc1=nn.Linear(conv_outp_size,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,hidden_dim)
        self.fc3=nn.Linear(hidden_dim,hidden_dim)
                
        self.output=nn.Linear(hidden_dim,action_dim)
        
        self.apply(self.init_weights)
        
    
    def forward(self,x):
        x=x/255
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))

        x=x.view(x.size(0),-1)

        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
    
        return self.output(x)
    
    
    # helper functions below this
    
    def calc_conv_output(self,observation_shape):
        x= torch.zeros(1,*observation_shape)
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))
        
        return x.view(-1).shape[0]
    
    
    def init_weights(self,m):
        if isinstance(m,nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias,0)
        elif isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias,0)
                
                
    def save_model(self,filename='models/latest.pt'):
        torch.save(self.state_dict(),filename)

    def load_model(self,filename='models/latest.pt'):
        # torch.save(self.state_dict(),filename)
        try:
            self.load_state_dict(torch.load(filename))
            print("loaded weights from ",filename)
            
        except FileNotFoundError:
            print("No weights found at " , filename)
            
            
def soft_update(target,source,tau=0.005):
    for target_param, param in zip(target.parameters(),source.parameters()):
        target_param.data.copy_(target_param.data*(1.0-tau)+param.data*tau)