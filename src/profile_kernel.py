import torch 
from binary import OptimizedBinaryLinear
from torch import nn

def profile_kernel():
    batch_size = 32
    in_features = 1024
    out_features = 768
    
    x = torch.randn(batch_size, in_features)
    layer = OptimizedBinaryLinear(in_features, out_features)
    layer.update_binary_weights()
    
    for _ in range(10):
        _ = layer(x)
    
    import time
    iterations = 100
    
    start = time.time()
    for _ in range(iterations):
        _ = layer(x)
    end = time.time()
    
    avg_time = (end - start) / iterations
    print(f"Average time per forward pass: {avg_time*1000:.4f} ms")
    print(f"Throughput: {batch_size / avg_time:.1f} samples/second")
    
    standard_layer = nn.Linear(in_features, out_features)
    
    start = time.time()
    for _ in range(iterations):
        _ = standard_layer(x)
    end = time.time()
    
    avg_time_standard = (end - start) / iterations
    print(f"Standard linear average time: {avg_time_standard*1000:.4f} ms")
    print(f"Speedup: {avg_time_standard / avg_time:.2f}x")

profile_kernel()