import torch
import torch.nn as nn
import torch.nn.functional as F

class IntEncoder(nn.Module):
    def __init__(self, max_size=torch.inf):
        super(IntEncoder, self).__init__()
        self._size = 0
        self._map = {}
        self.max_size = max_size

    def forward(self, x):
        
