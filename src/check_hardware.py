import torch
from model import get_hardware_info
import platform
import psutil

def get_system_info():
    """Get general system information"""
    return {
        'cpu_count': psutil.cpu_count(),
        'physical_cpu_count': psutil.cpu_count(logical=False),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'platform': platform.platform(),
        'python_version': platform.python_version()
    }

def check_hardware():
    """Check hardware capabilities and optimizations"""
    hw_info = get_hardware_info()
    sys_info = get_system_info()
    
    print("\nSystem Information:")
    print("-" * 50)
    print(f"Platform: {sys_info['platform']}")
    print(f"Python: {sys_info['python_version']}")
    print(f"CPU Cores (Physical/Logical): {sys_info['physical_cpu_count']}/{sys_info['cpu_count']}")
    print(f"Memory: {sys_info['memory_total'] / (1024**3):.1f}GB total, {sys_info['memory_available'] / (1024**3):.1f}GB available")
    
    print("\nHardware Detection Results:")
    print("-" * 50)
    
    # Device info
    print(f"Primary Device: {hw_info['device']}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")
            
            # Check tensor core capability
            gpu_cap = torch.cuda.get_device_capability(i)
            has_tensor_cores = gpu_cap[0] >= 7
            print(f"Tensor Cores Available: {has_tensor_cores}")
            print(f"GPU Compute Capability: {gpu_cap[0]}.{gpu_cap[1]}")
            print(f"Memory Total: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.1f}GB")
            print(f"Memory Reserved: {torch.cuda.memory_reserved(i) / (1024**3):.1f}GB")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(i) / (1024**3):.1f}GB")
    
    print("\nOptimized Training Settings:")
    print("-" * 50)
    print(f"Batch Size: {hw_info['optimal_batch_size']}")
    print(f"Worker Threads: {hw_info['optimal_workers']}")
    print(f"Mixed Precision Training: {hw_info['use_amp']}")
    
    if torch.cuda.is_available():
        print("\nCUDA Optimizations:")
        print("-" * 50)
        print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
        print(f"cuDNN Benchmark Mode: {torch.backends.cudnn.benchmark}")
        print(f"cuDNN Deterministic Mode: {torch.backends.cudnn.deterministic}")
        print(f"TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"cuBLAS TF32 Enabled: {torch.backends.cudnn.allow_tf32}")

if __name__ == "__main__":
    check_hardware()