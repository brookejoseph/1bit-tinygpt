import ctypes
import numpy as np
import torch

lib = ctypes.CDLL('./liboptimized_apple.so')

lib.binary_matmul_apple_m.argtypes = [
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int                      
]

def binary_matmul(input_tensor, packed_weights, scaling_factors):
    """Python wrapper for the optimized binary matrix multiplication"""
    batch_size, in_features = input_tensor.shape
    out_features = scaling_factors.shape[0]
    
    output = torch.zeros(batch_size, out_features, dtype=torch.float32)
    
    input_np = input_tensor.numpy().astype(np.float32)
    weights_np = packed_weights.numpy().astype(np.uint32)
    scaling_np = scaling_factors.detach().numpy().astype(np.float32)
    output_np = output.numpy()
    
    input_ptr = input_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    weights_ptr = weights_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    scaling_ptr = scaling_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    lib.binary_matmul_apple_m(
        weights_ptr, scaling_ptr, input_ptr, output_ptr,
        batch_size, in_features, out_features
    )
    
    return torch.from_numpy(output_np)