import torch
import math 
import bindings


from torch import nn

class OptimizedBinaryLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.scaling = nn.Parameter(torch.ones(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        ints_per_row = (in_features + 31) // 32
        self.register_buffer('packed_weights', 
                           torch.zeros(out_features, ints_per_row, dtype=torch.int32))
        
    def update_binary_weights(self):
        """Update packed binary weights from full-precision weights"""
        sign_weights = torch.sign(self.weight)
        sign_weights[sign_weights == 0] = 1  
        
        self.scaling.data = torch.mean(torch.abs(self.weight), dim=1)
        
        out_f, in_f = self.weight.shape
        ints_per_row = (in_f + 31) // 32
        
        bits = (sign_weights + 1) // 2
        
        self.packed_weights.zero_()
        
        for i in range(out_f):
            for j in range(in_f):
                if bits[i, j]:
                    self.packed_weights[i, j // 32] |= (1 << (j % 32))
    
    def forward(self, x):
        self.update_binary_weights()
        
        output = bindings.binary_matmul(x, self.packed_weights, self.scaling.view(-1, 1))
        
        if self.bias is not None:
            output = output + self.bias
            
        return output