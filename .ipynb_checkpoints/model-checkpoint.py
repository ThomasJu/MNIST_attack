import torch
from torch import nn
from torch.nn import functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p=0.25)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.l1 = nn.LazyConv2d(32, 5)  # l2 regularization?
        self.l2 = nn.LazyConv2d(32, 5, bias=False)
        self.l3 = nn.LazyConv2d(64, 3)  # l2 regularization?
        self.l4 = nn.LazyConv2d(64, 3, bias=False) 
        self.l5 = nn.LazyLinear(256, bias=False)
        self.l6 = nn.LazyLinear(128, bias=False)
        self.l7 = nn.LazyLinear(64, bias=False)
        self.l8 = nn.LazyLinear(10, bias=False)
    
    def forward(self, x):
#         x = F.relu(self.l1(x))
#         x = F.relu(self.l2(x))
#         x = self.dropout(self.maxpool(x))
#         x = F.relu(self.l3(x))
#         x = F.relu(self.l4(x))
#         x = self.dropout(self.maxpool(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.l5(x))
        x = F.relu(self.l6(x))
        x = F.relu(self.l7(x))
        x = F.softmax(self.l8(x))
        
        return x