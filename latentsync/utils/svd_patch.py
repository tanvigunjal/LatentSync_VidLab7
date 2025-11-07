"""Patch for optimized SVD operations on MPS."""

import torch

def optimized_svd(tensor):
    """Optimized SVD implementation for MPS devices."""
    if tensor.device.type == "mps":
        # Move to CPU, perform SVD, and move back to MPS
        with torch.no_grad():
            cpu_tensor = tensor.cpu()
            U, S, V = torch.svd(cpu_tensor)
            return U.to("mps"), S.to("mps"), V.to("mps")
    else:
        return torch.svd(tensor)

# Monkey patch torch.svd for MPS devices
original_svd = torch.svd

def patched_svd(tensor, *args, **kwargs):
    if tensor.device.type == "mps":
        return optimized_svd(tensor)
    return original_svd(tensor, *args, **kwargs)

# Apply the patch
torch.svd = patched_svd