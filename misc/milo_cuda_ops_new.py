"""
CUDA-accelerated operations for MILO optimizer.
Uses PyTorch's built-in CUDA operations for maximum compatibility.
"""

import torch

# Set availability flag
CUDA_AVAILABLE = torch.cuda.is_available()

def cuda_gradient_normalization(gradients, eps=1e-5, scale_factor=0.2, scale_aware=True):
    """CUDA-accelerated gradient normalization using PyTorch operations."""
    if not CUDA_AVAILABLE or not gradients.is_cuda:
        return gradients
    
    grad_norm = torch.norm(gradients)
    
    if grad_norm > eps:
        normalization_factor = 1.0 / grad_norm
        
        if scale_aware:
            clamped_norm = grad_norm.clamp(max=1.0)
            normalized_grad = gradients * normalization_factor
            return gradients * scale_factor + normalized_grad * (1 - scale_factor) * clamped_norm
        else:
            return gradients * normalization_factor
    
    return gradients

def cuda_enhanced_momentum(momentum_long, momentum_short, gradients, 
                          momentum_long_decay, momentum_short_decay):
    """CUDA-accelerated enhanced momentum update using PyTorch operations."""
    if not CUDA_AVAILABLE or not gradients.is_cuda:
        return momentum_long, momentum_short
    
    momentum_long.mul_(momentum_long_decay).add_(gradients, alpha=1-momentum_long_decay)
    momentum_short.mul_(momentum_short_decay).add_(gradients, alpha=1-momentum_short_decay)
    
    return momentum_long, momentum_short

def cuda_layerwise_normalization(param_grads, eps=1e-5, scale_factor=0.2, scale_aware=True):
    """CUDA-accelerated layer-wise gradient normalization."""
    if not param_grads or not CUDA_AVAILABLE:
        return
    
    flat_grads = [grad.flatten() for grad in param_grads if grad.is_cuda]
    if not flat_grads:
        return
    
    all_grads = torch.cat(flat_grads)
    grad_norm = torch.norm(all_grads)
    
    if grad_norm > eps:
        normalization_factor = 1.0 / grad_norm
        
        for grad in param_grads:
            if not grad.is_cuda:
                continue
                
            grad_flat = grad.view(-1)
            
            if scale_aware:
                clamped_norm = grad_norm.clamp(max=1.0)
                normalized_component = grad_flat * normalization_factor
                grad_flat.mul_(scale_factor).add_(normalized_component, 
                              alpha=(1 - scale_factor) * clamped_norm.item())
            else:
                grad_flat.mul_(normalization_factor)

# PyTorch fallback functions
def torch_gradient_normalization(gradients, eps=1e-5, scale_factor=0.2, scale_aware=True):
    """Fallback gradient normalization using PyTorch operations."""
    grad_norm = torch.norm(gradients)
    
    if grad_norm > eps:
        normalization_factor = 1.0 / grad_norm
        
        if scale_aware:
            clamped_norm = grad_norm.clamp(max=1.0)
            normalized_grad = gradients * normalization_factor
            return gradients * scale_factor + normalized_grad * (1 - scale_factor) * clamped_norm
        else:
            return gradients * normalization_factor
    
    return gradients

def torch_enhanced_momentum(momentum_long, momentum_short, gradients, 
                           momentum_long_decay, momentum_short_decay):
    """Fallback enhanced momentum update using PyTorch operations."""
    momentum_long.mul_(momentum_long_decay).add_(gradients, alpha=1-momentum_long_decay)
    momentum_short.mul_(momentum_short_decay).add_(gradients, alpha=1-momentum_short_decay)
    return momentum_long, momentum_short

def torch_layerwise_normalization(param_grads, eps=1e-5, scale_factor=0.2, scale_aware=True):
    """Fallback layer-wise gradient normalization."""
    if not param_grads:
        return
    
    flat_grads = [grad.flatten() for grad in param_grads]
    all_grads = torch.cat(flat_grads)
    grad_norm = torch.norm(all_grads)
    
    if grad_norm > eps:
        normalization_factor = 1.0 / grad_norm
        
        for grad in param_grads:
            grad_flat = grad.view(-1)
            
            if scale_aware:
                clamped_norm = grad_norm.clamp(max=1.0)
                normalized_component = grad_flat * normalization_factor
                alpha_value = ((1 - scale_factor) * clamped_norm).item()
                grad_flat.mul_(scale_factor).add_(normalized_component, alpha=alpha_value)
            else:
                grad_flat.mul_(normalization_factor)

# Create a compatible interface for existing code
class CudaOps:
    """CUDA operations interface using PyTorch built-in CUDA support."""
    
    def fused_gradient_normalization(self, gradients, eps=1e-5, scale_factor=0.2, scale_aware=True):
        if CUDA_AVAILABLE and gradients.is_cuda:
            return cuda_gradient_normalization(gradients, eps, scale_factor, scale_aware)
        else:
            return torch_gradient_normalization(gradients, eps, scale_factor, scale_aware)
    
    def enhanced_momentum_update(self, momentum_long, momentum_short, gradients, 
                                momentum_long_decay, momentum_short_decay):
        if CUDA_AVAILABLE and gradients.is_cuda:
            return cuda_enhanced_momentum(momentum_long, momentum_short, gradients, 
                                        momentum_long_decay, momentum_short_decay)
        else:
            return torch_enhanced_momentum(momentum_long, momentum_short, gradients,
                                         momentum_long_decay, momentum_short_decay)
    
    def layerwise_normalization(self, param_grads, eps=1e-5, scale_factor=0.2, scale_aware=True):
        if CUDA_AVAILABLE and any(grad.is_cuda for grad in param_grads):
            cuda_layerwise_normalization(param_grads, eps, scale_factor, scale_aware)
        else:
            torch_layerwise_normalization(param_grads, eps, scale_factor, scale_aware)

# Create the milo_cuda object for compatibility
milo_cuda = CudaOps()

print(f"MILO CUDA acceleration ready: {CUDA_AVAILABLE}")
if CUDA_AVAILABLE:
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    print(f"Using GPU: {device_name} (Device {current_device}/{device_count})")
