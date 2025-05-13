import torch

def configure_h200_environment():
    """Configure environment for optimal H200 performance"""
    # Use torch.compile for performance (requires PyTorch 2.0+)
    torch.backends.cudnn.benchmark = True
    
    # Set environment variables for H200 performance
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    # Optional: Enable TF32 for faster computation on H100/H200
    torch.set_float32_matmul_precision('high')
    
    # Set CUDA device
    torch.cuda.set_device(0)