import torch
import torch.nn as nn
import torch.nn.functional as F

class IntEncoder(nn.Module):
    def __init__(self, input_shape, max_size=torch.inf):
        super(IntEncoder, self).__init__()
        self._size = 0
        self._map = {}
        self.input_shape = input_shape
        self.max_size = max_size

    def forward(self, x):
        if x.shape == self.input_shape:
            return self.encode_single(x)
        elif x.shape[1:] == self.input_shape:
            return self.encode_multiple(x)
        else:
            raise Exception(f"Input doesn't match shape {self.input_shape}")
            
        
    def encode_single(self, x):
        key =  tuple(x.ravel().tolist())
        if key in self._map:
            return self._map[key]
        elif self._size == self.max_size:
            raise Exception("Encoder ran out of unique mappings")
        else:
            self._map[key] = self._size
            self._size += 1
            return self._map[key]
        
    def encode_multiple(self, arr):
        r = torch.tensor(dtype=torch.int64)
        for x in arr:
            r.append(self.encode(x))
        return r