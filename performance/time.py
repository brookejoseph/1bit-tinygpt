import time
import torch
import torch.profiler as profiler
from transformers import AutoModelForCausalLM

class ModelProfiler:
    def __init__(self, model, input_ids):
        self.model = model
        self.input_ids = input_ids
    
    def profile_all(self):
        """Run all profiling methods in sequence"""
        print("\n=== GENERATION TIME ===")
        self.profile_generation_time()
        
        print("\n=== MATRIX MULTIPLICATIONS ===")
        self.profile_matmuls()
        
        print("\n=== LAYER BREAKDOWN ===")
        self.profile_layers()
        
        print("\n=== PYTORCH PROFILER RESULTS ===")
        self.profile_with_pytorch()
        
        print("\n=== MEMORY USAGE ===")
        self.profile_memory()
    
    def profile_generation_time(self):
        """Measure total time for text generation"""
        start_time = time.time()
        output = self.model.generate(self.input_ids.input_ids, max_length=50)
        end_time = time.time()
        print(f"Total generation took {end_time - start_time:.4f} seconds")
        return end_time - start_time

    def profile_with_pytorch(self):
        """Use PyTorch's built-in profiler for detailed timing"""
        with profiler.profile(
            activities=[profiler.ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            self.model(self.input_ids.input_ids)
        
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        return prof

    def profile_layers(self):
        """Profile individual transformer blocks and their components"""
        sample = self.input_ids.input_ids
        layer_times = []
        
        for i, block in enumerate(self.model.transformer.h):
            with torch.no_grad():
                if i == 0:
                    hidden_states = self.model.transformer.wte(sample)
                else:
                    pass
            
            start = time.time()
            with torch.no_grad():
                output = block(hidden_states)
            end = time.time()
            block_time = (end - start) * 1000
            print(f"Block {i} took {block_time:.2f} ms")
            
            start = time.time()
            with torch.no_grad():
                attn_output = block.attn(block.ln_1(hidden_states))
            end = time.time()
            attn_time = (end - start) * 1000
            print(f"  - Attention in Block {i}: {attn_time:.2f} ms")
            
            start = time.time()
            with torch.no_grad():
                ffn_output = block.mlp(block.ln_2(hidden_states))
            end = time.time()
            ffn_time = (end - start) * 1000
            print(f"  - Feed-forward in Block {i}: {ffn_time:.2f} ms")
            
            layer_times.append((i, block_time, attn_time, ffn_time))
        
        return layer_times

    def profile_matmuls(self):
        """Focus on matrix multiplications (critical for BitNet)"""
        original_matmul = torch.matmul
        
        matmul_count = 0
        matmul_times = []
        
        def timed_matmul(a, b):
            nonlocal matmul_count
            start = time.time()
            result = original_matmul(a, b)
            end = time.time()
            
            if a.numel() > 1000 and b.numel() > 1000:
                matmul_count += 1
                matmul_times.append((end - start, a.shape, b.shape))
            
            return result
        
        torch.matmul = timed_matmul
        
        with torch.no_grad():
            self.model(self.input_ids.input_ids)
        
        torch.matmul = original_matmul
        
        total_time = sum(t[0] for t in matmul_times)
        print(f"Found {matmul_count} significant matrix multiplications")
        print(f"Total matmul time: {total_time*1000:.2f} ms")
        
        sorted_times = sorted(matmul_times, key=lambda x: x[0], reverse=True)
        for i, (t, shape_a, shape_b) in enumerate(sorted_times[:5]):
            print(f"MatMul #{i+1}: {t*1000:.2f} ms, shapes: {shape_a} × {shape_b}")
        
        return matmul_times

    def profile_memory(self):
        """Profile memory usage of the model"""
        memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        with torch.no_grad():
            output = self.model(self.input_ids.input_ids)
        
        memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        memory_used = memory_after - memory_before
        print(f"Memory used for forward pass: {memory_used / 1024 / 1024:.2f} MB")
        
        model_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        print(f"Model parameter size: {model_size / 1024 / 1024:.2f} MB")
        
        return {"used": memory_used, "parameters": model_size}


if __name__ == "__main__":
    from loading_model import grab_model_info
    
    tokenizer, input_ids, model = grab_model_info()
    
    profiler = ModelProfiler(model, input_ids)
    profiler.profile_all()