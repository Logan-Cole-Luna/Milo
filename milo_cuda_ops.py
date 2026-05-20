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
    """CUDA-accelerated layer-wise gradient normalization without concatenation."""
    if not param_grads or not CUDA_AVAILABLE:
        return

    # Two-pass norm accumulation to avoid large temporary tensors
    device = None
    sq_sum = None
    for grad in param_grads:
        if grad is None or not isinstance(grad, torch.Tensor) or not grad.is_cuda:
            continue
        if device is None:
            device = grad.device
            sq_sum = torch.zeros(1, device=device, dtype=torch.float32)
        sq_sum.add_(grad.detach().to(torch.float32).pow(2).sum())

    if sq_sum is None:
        return

    grad_norm = sq_sum.sqrt()
    # If norm is tiny, skip to avoid numerical issues
    if (grad_norm <= eps).item():
        return

    normalization_factor = grad_norm.reciprocal()  # stay on device
    clamped_norm = grad_norm.clamp(max=1.0)

    # Filter CUDA tensors and ensure homogeneity for foreach
    grads_cuda = [g for g in param_grads if isinstance(g, torch.Tensor) and g.is_cuda]
    if not grads_cuda:
        return
    same_kind = len({(g.device, g.dtype) for g in grads_cuda}) == 1 and len(grads_cuda) > 1

    if same_kind:
        if scale_aware:
            normalized = torch._foreach_mul(grads_cuda, normalization_factor)
            torch._foreach_mul_(grads_cuda, scale_factor)
            torch._foreach_add_(grads_cuda, normalized, alpha=(1 - scale_factor) * clamped_norm)
        else:
            torch._foreach_mul_(grads_cuda, normalization_factor)
    else:
        for grad in grads_cuda:
            if scale_aware:
                normalized_term = (grad * normalization_factor) * clamped_norm
                grad.mul_(scale_factor)
                grad.add_(normalized_term, alpha=(1 - scale_factor))
            else:
                grad.mul_(normalization_factor)

def cuda_parameterwise_batch_normalization(param_grads, eps=1e-5, scale_factor=0.2, scale_aware=True):
    """CUDA-accelerated parameter-wise normalization in-place using foreach ops.

    Applies per-tensor normalization factors without creating new tensors, batching
    operations across tensors of the same device/dtype. Leaves gradients with tiny
    norms (<= eps) unchanged to match existing semantics.
    """
    if not CUDA_AVAILABLE or not param_grads:
        return

    # Group by (device, dtype) and keep only CUDA tensors
    groups = {}
    for g in param_grads:
        if not isinstance(g, torch.Tensor) or not g.is_cuda:
            continue
        key = (g.device, g.dtype)
        groups.setdefault(key, []).append(g)

    for (device, dtype), grads_cuda in groups.items():
        if len(grads_cuda) == 0:
            continue

        # Compute norms per grad on device (no host sync yet)
        norms = [torch.linalg.vector_norm(g) for g in grads_cuda]
        norms_tensor = torch.stack(norms)  # shape [N], device: CUDA

        # Determine which grads to skip (tiny norms)
        small_mask = norms_tensor <= eps

        if scale_aware:
            clamped = norms_tensor.clamp_max(1.0)
            # inv_norm = 1/norm, safe where small -> unused
            inv_norm = torch.where(small_mask, torch.ones_like(norms_tensor), norms_tensor.reciprocal())
            scales = torch.where(
                small_mask,
                torch.ones_like(norms_tensor),
                torch.as_tensor(scale_factor, device=device, dtype=norms_tensor.dtype) +
                (1 - scale_factor) * inv_norm * clamped,
            )
        else:
            inv_norm = torch.where(small_mask, torch.ones_like(norms_tensor), norms_tensor.reciprocal())
            scales = torch.where(small_mask, torch.ones_like(norms_tensor), inv_norm)

        # foreach expects Python scalars per tensor; move to CPU as list of floats
        scales_list = scales.tolist()  # triggers a small host sync per group

        # In-place multiply each grad by its scale in one batched op
        torch._foreach_mul_(grads_cuda, scales_list)

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

    def parameterwise_batch_normalization(self, param_grads, eps=1e-5, scale_factor=0.2, scale_aware=True):
        if CUDA_AVAILABLE and any(isinstance(g, torch.Tensor) and g.is_cuda for g in param_grads):
            cuda_parameterwise_batch_normalization(param_grads, eps, scale_factor, scale_aware)
        else:
            # Fallback: CPU foreach when homogeneous, else loop
            grads_cpu = [g for g in param_grads if isinstance(g, torch.Tensor) and not g.is_cuda]
            if len(grads_cpu) == 0:
                return
            # Group by dtype for safety
            groups = {}
            for g in grads_cpu:
                groups.setdefault(g.dtype, []).append(g)
            for dtype, grads in groups.items():
                norms = [torch.norm(g) for g in grads]
                norms_tensor = torch.stack(norms)
                small_mask = norms_tensor <= eps
                if scale_aware:
                    clamped = norms_tensor.clamp_max(1.0)
                    inv_norm = torch.where(small_mask, torch.ones_like(norms_tensor), norms_tensor.reciprocal())
                    scales = torch.where(
                        small_mask,
                        torch.ones_like(norms_tensor),
                        torch.as_tensor(scale_factor, dtype=norms_tensor.dtype) + (1 - scale_factor) * inv_norm * clamped,
                    )
                else:
                    inv_norm = torch.where(small_mask, torch.ones_like(norms_tensor), norms_tensor.reciprocal())
                    scales = torch.where(small_mask, torch.ones_like(norms_tensor), inv_norm)
                torch._foreach_mul_(grads, scales.tolist())

# Create the milo_cuda object for compatibility
milo_cuda = CudaOps()

# Backward-compatibility shim: expose MiloCudaOps with static methods to satisfy imports/tests
class MiloCudaOps:
    """
    Compatibility class providing static methods expected by older tests.
    Methods delegate to the lightweight PyTorch CUDA implementations above.
    """

    @staticmethod
    def fused_gradient_normalization(
        gradients,
        param_sizes=None,
        param_offsets=None,
        layer_indices=None,
        eps=1e-5,
        scale_factor=0.2,
        scale_aware=True,
        layer_wise=True,
    ):
        # Accept either a single tensor or a list of tensors (use first element)
        if isinstance(gradients, (list, tuple)):
            if len(gradients) == 0:
                return gradients
            grad = gradients[0]
        else:
            grad = gradients

        if CUDA_AVAILABLE and isinstance(grad, torch.Tensor) and grad.is_cuda:
            return cuda_gradient_normalization(grad, eps=eps, scale_factor=scale_factor, scale_aware=scale_aware)
        else:
            return torch_gradient_normalization(grad, eps=eps, scale_factor=scale_factor, scale_aware=scale_aware)

    @staticmethod
    def enhanced_momentum_update(momentum_long, momentum_short, gradients, momentum_long_decay, momentum_short_decay):
        if CUDA_AVAILABLE and isinstance(gradients, torch.Tensor) and gradients.is_cuda:
            ml, ms = cuda_enhanced_momentum(momentum_long, momentum_short, gradients, momentum_long_decay, momentum_short_decay)
            return ml
        else:
            ml, ms = torch_enhanced_momentum(momentum_long, momentum_short, gradients, momentum_long_decay, momentum_short_decay)
            return ml

print(f"MILO CUDA acceleration ready: {CUDA_AVAILABLE}")
if CUDA_AVAILABLE:
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    print(f"Using GPU: {device_name} (Device {current_device}/{device_count})")
