import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class MarketCNN(nn.Module):
  
    def __init__(self, emb_dim: int = 768):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1,  out_channels=14, kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(in_channels=14, out_channels=64, kernel_size=3, stride=1)
        conv_out_dim = ((emb_dim - 2 - 2) // 2) * 64

        self.fc1 = nn.Linear(conv_out_dim, 128)
        self.fc2 = nn.Linear(128, 1)          

    def forward(self, x):
        x = x.unsqueeze(1)                    
        x = F.relu(self.conv1(x))             
        x = F.relu(self.conv2(x))             
        x = F.max_pool1d(x, 2)              
        x = torch.flatten(x, 1)              
        x = F.relu(self.fc1(x))              
        x = self.fc2(x)                      
        return x.squeeze(1)                  