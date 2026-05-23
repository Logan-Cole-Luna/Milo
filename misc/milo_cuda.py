# mypy: allow-untyped-defs
r"""MILO v1 Enhanced with CUDA kernel optimizations."""
import math
import time
from typing import cast, List, Optional, Union

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT

# Import CUDA operations
try:
    from milo_cuda_ops import MiloCudaOps
    CUDA_OPS_AVAILABLE = True
except ImportError:
    print("Warning: CUDA operations not available, falling back to standard PyTorch")
    CUDA_OPS_AVAILABLE = False
    MiloCudaOps = None

__all__ = ["milo_cuda"]


class milo_cuda(Optimizer):  # noqa: D101
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        momentum: float = 0.9,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        differentiable: bool = False,
        # Parameters for gradient normalization
        normalize: bool = True,
        layer_wise: bool = True,
        group_size: Optional[int] = None,
        eps: float = 1e-5,
        scale_aware: bool = True,
        scale_factor: float = 0.2,
        max_group_size: Optional[int] = 5000,
        clip_norm: Optional[float] = None,
        adaptive: bool = True,
        adaptive_eps: float = 1e-8,
        disable_layer_mapping: bool = False,
        profile_time: bool = False,
        use_cached_mapping: bool = False,
        layer_lr_multipliers: Optional[dict] = None,
        # CUDA optimization settings
        use_cuda_kernels: bool = True,
        force_cuda_fallback: bool = False,
    ):
        """
        MILO v1 Enhanced with CUDA kernel optimizations for gradient normalization.
        
        This version provides significant performance improvements through fused CUDA kernels
        that combine multiple gradient operations into single GPU kernel launches.
        
        New CUDA Features:
        - Fused gradient normalization kernel
        - Fused momentum and parameter update kernel
        - Automatic fallback to PyTorch operations when CUDA unavailable
        
        Args:
            # ... (existing MILO args) ...
            use_cuda_kernels (bool): Enable CUDA kernel optimizations when available
            force_cuda_fallback (bool): Force fallback to PyTorch operations (for testing)
        """
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
            foreach=foreach,
            differentiable=differentiable,
            # Normalization parameters
            normalize=normalize,
            layer_wise=layer_wise,
            group_size=group_size,
            eps=eps,
            scale_aware=scale_aware,
            scale_factor=scale_factor,
            max_group_size=max_group_size,
            clip_norm=clip_norm,
            adaptive=adaptive,
            adaptive_eps=adaptive_eps,
            disable_layer_mapping=disable_layer_mapping,
            profile_time=profile_time,
            use_cached_mapping=use_cached_mapping,
            layer_lr_multipliers=layer_lr_multipliers or {},
            # CUDA settings
            use_cuda_kernels=use_cuda_kernels,
            force_cuda_fallback=force_cuda_fallback,
        )
        super().__init__(params, defaults)

        # Initialize layer mapping and CUDA optimization state
        self.param_to_layer = {}
        self._cached_buffers = {}
        self.profile_stats = {"normalize_time": 0, "sgd_time": 0, "cuda_time": 0, "total_steps": 0}
        self._cuda_available = CUDA_OPS_AVAILABLE and not force_cuda_fallback
        
        if use_cached_mapping:
            self._create_layer_mapping()
            
        # Initialize CUDA optimization metadata
        if self._cuda_available:
            self._init_cuda_metadata()

    def _create_layer_mapping(self):
        """Create mapping from parameters to layer indices."""
        layer_counter = 0
        for group in self.param_groups:
            for i, param in enumerate(group["params"]):
                if hasattr(param, '_param_name'):
                    layer_name = param._param_name.split('.')[0] if '.' in param._param_name else f"layer_{layer_counter}"
                    self.param_to_layer[param] = layer_name
                else:
                    self.param_to_layer[param] = f"layer_{layer_counter}"
                    layer_counter += 1

    def _init_cuda_metadata(self):
        """Initialize metadata needed for CUDA kernels."""
        self._cuda_metadata = {}
        
        for group in self.param_groups:
            if group["layer_wise"]:
                # Prepare layer-wise normalization metadata
                layer_params = {}
                for param in group["params"]:
                    layer_idx = self.param_to_layer.get(param, 0)
                    if layer_idx not in layer_params:
                        layer_params[layer_idx] = []
                    layer_params[layer_idx].append(param)
                
                # Create tensors for CUDA kernel inputs
                param_sizes = []
                param_offsets = []
                layer_indices = []
                offset = 0
                
                for layer_idx, params in layer_params.items():
                    for param in params:
                        param_sizes.append(param.numel())
                        param_offsets.append(offset)
                        layer_indices.append(layer_idx)
                        offset += param.numel()
                
                self._cuda_metadata[id(group)] = {
                    'param_sizes': torch.tensor(param_sizes, dtype=torch.int32),
                    'param_offsets': torch.tensor(param_offsets, dtype=torch.int32),
                    'layer_indices': torch.tensor(layer_indices, dtype=torch.int32),
                    'layer_params': layer_params
                }

    def step(self, closure=None):
        """Perform a single optimization step with CUDA kernel optimizations."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        start_time = time.time() if any(group["profile_time"] for group in self.param_groups) else None

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            momentum_buffers = []
            
            # Collect parameters with gradients
            for param in group["params"]:
                if param.grad is not None:
                    params_with_grad.append(param)
                    grads.append(param.grad)
                    
                    state = self.state[param]
                    if len(state) == 0:
                        state['momentum_buffer'] = torch.zeros_like(param)
                        if group["adaptive"]:
                            state['exp_avg_sq'] = torch.zeros_like(param)
                    
                    momentum_buffers.append(state)

            if not params_with_grad:
                continue

            # Apply gradient normalization with CUDA acceleration
            if group["normalize"]:
                norm_start = time.time() if group["profile_time"] else None
                self._normalize_gradients_cuda(group, params_with_grad, grads)
                if norm_start and group["profile_time"]:
                    self.profile_stats["normalize_time"] += time.time() - norm_start

            # Apply parameter updates with CUDA acceleration
            sgd_start = time.time() if group["profile_time"] else None
            self._apply_updates_cuda(group, params_with_grad, grads, momentum_buffers)
            if sgd_start and group["profile_time"]:
                self.profile_stats["sgd_time"] += time.time() - sgd_start

        # Update profiling stats
        if start_time:
            self.profile_stats["total_steps"] += 1
            if self.profile_stats["total_steps"] % 100 == 0:
                cuda_status = "CUDA" if self._cuda_available else "PyTorch"
                print(f"MILO-CUDA Profiling ({cuda_status}) - "
                      f"Normalization: {self.profile_stats['normalize_time']/self.profile_stats['total_steps']*1000:.2f}ms/step, "
                      f"Updates: {self.profile_stats['sgd_time']/self.profile_stats['total_steps']*1000:.2f}ms/step")

        return loss

    def _normalize_gradients_cuda(self, group, params, grads):
        """Apply gradient normalization using CUDA kernels when available."""
        if not group["normalize"] or not grads:
            return

        cuda_start = time.time() if group["profile_time"] else None
        
        # Check if we can use CUDA acceleration
        use_cuda = (self._cuda_available and 
                   group["use_cuda_kernels"] and 
                   not group.get("force_cuda_fallback", False) and
                   all(g.is_cuda for g in grads))

        if use_cuda and MiloCudaOps is not None:
            try:
                # Use CUDA fused kernel
                group_id = id(group)
                if group_id in self._cuda_metadata:
                    metadata = self._cuda_metadata[group_id]
                    MiloCudaOps.fused_gradient_normalization(
                        grads,
                        metadata['param_sizes'].to(grads[0].device),
                        metadata['param_offsets'].to(grads[0].device),
                        metadata['layer_indices'].to(grads[0].device),
                        eps=group["eps"],
                        scale_factor=group["scale_factor"],
                        scale_aware=group["scale_aware"],
                        layer_wise=group["layer_wise"]
                    )
                else:
                    # Fallback for non-layer-wise or first-time setup
                    self._fallback_normalize_gradients(group, params, grads)
            except Exception as e:
                print(f"Warning: CUDA kernel failed ({e}), falling back to PyTorch")
                self._fallback_normalize_gradients(group, params, grads)
        else:
            # Use PyTorch fallback
            self._fallback_normalize_gradients(group, params, grads)

        if cuda_start and group["profile_time"]:
            self.profile_stats["cuda_time"] += time.time() - cuda_start

    def _apply_updates_cuda(self, group, params, grads, momentum_buffers):
        """Apply parameter updates using CUDA kernels when available."""
        use_cuda = (self._cuda_available and 
                   group["use_cuda_kernels"] and 
                   not group.get("force_cuda_fallback", False) and
                   all(p.is_cuda for p in params))

        if use_cuda and MiloCudaOps is not None:
            try:
                # Prepare buffers for CUDA kernel
                momentum_bufs = [state['momentum_buffer'] for state in momentum_buffers]
                exp_avg_sq_bufs = [state.get('exp_avg_sq', torch.zeros_like(p)) for p, state in zip(params, momentum_buffers)]
                
                # Apply weight decay if needed
                if group["weight_decay"] != 0:
                    for grad, param in zip(grads, params):
                        grad.add_(param, alpha=group["weight_decay"])

                # Use fused CUDA kernel
                MiloCudaOps.fused_momentum_adaptive_update(
                    params, grads, momentum_bufs, exp_avg_sq_bufs,
                    momentum=group["momentum"],
                    beta2=0.999,  # For adaptive learning rate
                    eps=group["adaptive_eps"],
                    lr=group["lr"],
                    use_adaptive_lr=group["adaptive"],
                    use_momentum=group["momentum"] > 0
                )
                
            except Exception as e:
                print(f"Warning: CUDA update kernel failed ({e}), falling back to PyTorch")
                self._fallback_apply_updates(group, params, grads, momentum_buffers)
        else:
            # Use PyTorch fallback
            self._fallback_apply_updates(group, params, grads, momentum_buffers)

    def _fallback_normalize_gradients(self, group, params, grads):
        """Fallback gradient normalization using PyTorch operations."""
        layer_wise = group["layer_wise"]
        disable_layer_mapping = group.get("disable_layer_mapping", False)
        
        if disable_layer_mapping or (layer_wise and not self.param_to_layer):
            for param, grad in zip(params, grads):
                self._normalize_single_param(group, grad)
            return
            
        if layer_wise:
            layer_params_dict = {}
            for param, grad in zip(params, grads):
                layer_idx = self.param_to_layer.get(param, 0)
                if layer_idx not in layer_params_dict:
                    layer_params_dict[layer_idx] = {"grads": []}
                layer_params_dict[layer_idx]["grads"].append(grad)
            
            for layer_idx, layer_data in layer_params_dict.items():
                self._normalize_layer_grads(group, layer_data["grads"])
        else:
            group_size = group["group_size"] or len(grads)
            for i in range(0, len(grads), group_size):
                group_grads = grads[i:i+group_size]
                self._normalize_layer_grads(group, group_grads)

    def _fallback_apply_updates(self, group, params, grads, momentum_buffers):
        """Fallback parameter updates using PyTorch operations."""
        for param, grad, state in zip(params, grads, momentum_buffers):
            # Apply weight decay
            if group["weight_decay"] != 0:
                grad = grad.add(param, alpha=group["weight_decay"])

            # Update momentum buffer
            momentum_buffer = state['momentum_buffer']
            if group["momentum"] != 0:
                momentum_buffer.mul_(group["momentum"]).add_(grad, alpha=1-group["dampening"])
                if group["nesterov"]:
                    grad = grad.add(momentum_buffer, alpha=group["momentum"])
                else:
                    grad = momentum_buffer

            # Apply adaptive learning rate if enabled
            if group["adaptive"]:
                exp_avg_sq = state['exp_avg_sq']
                exp_avg_sq.mul_(0.999).addcmul_(grad, grad, value=1-0.999)
                adaptive_lr = group["lr"] / (exp_avg_sq.sqrt() + group["adaptive_eps"])
                param.data.add_(grad * adaptive_lr, alpha=-1)
            else:
                param.data.add_(grad, alpha=-group["lr"])

    def _normalize_single_param(self, group, grad):
        """Normalize a single parameter's gradient."""
        eps = group["eps"]
        scale_aware = group["scale_aware"]
        scale_factor = group["scale_factor"]
        
        grad_norm = torch.norm(grad)
        if grad_norm > eps:
            normalized_grad = grad / grad_norm
            if scale_aware:
                grad.copy_(scale_factor * grad + (1 - scale_factor) * normalized_grad * grad_norm.clamp(max=1.0))
            else:
                grad.copy_(normalized_grad)

    def _normalize_layer_grads(self, group, grads):
        """Normalize gradients within a layer."""
        if not grads:
            return
            
        eps = group["eps"]
        scale_aware = group["scale_aware"]
        scale_factor = group["scale_factor"]
        
        flat_grads = []
        for grad in grads:
            flat_grads.append(grad.flatten())
        
        all_grads = torch.cat(flat_grads)
        grad_norm = torch.norm(all_grads)
        
        if grad_norm > eps:
            normalization_factor = 1.0 / grad_norm
            
            for grad in grads:
                if scale_aware:
                    normalized_grad = grad * normalization_factor
                    grad.copy_(scale_factor * grad + (1 - scale_factor) * normalized_grad * grad_norm.clamp(max=1.0))
                else:
                    grad.mul_(normalization_factor)
