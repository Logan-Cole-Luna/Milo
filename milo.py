# mypy: allow-untyped-defs
r"""Implementation for Normalized Stochastic Gradient Descent optimizer."""
import math
import time
from typing import cast, List, Optional, Union

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT

# Import CUDA operations
# TODO
CUDA_OPS_AVAILABLE = True

__all__ = ["milo"]


class milo(Optimizer):  # noqa: D101
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        momentum: float = 0,
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
        layer_lr_multipliers: Optional[dict] = None,  # Added for per-layer LR
        # CUDA optimization settings
        use_cuda_kernels: bool = True,
        force_cuda_fallback: bool = False,
        # Performance tuning
        normalize_interval: int = 1,
    ):
        """
        Implements Normalized Stochastic Gradient Descent (optionally with momentum).
        
        This optimizer extends standard SGD by normalizing gradients within groups of parameters,
        helping to balance the scale of updates across different layers of the network.
        
        Args:
            params (iterable): Iterable of parameters to optimize or dicts defining parameter groups
            lr (float, Tensor): Learning rate
            momentum (float): Momentum factor
            dampening (float): Dampening for momentum
            weight_decay (float): Weight decay (L2 penalty)
            nesterov (bool): Enables Nesterov momentum
            maximize (bool): Maximize the params based on the objective, instead of minimizing
            foreach (bool, optional): Whether to use foreach implementation of optimizer
            differentiable (bool): Whether to create differentiable optimizer
            normalize (bool): Enable gradient normalization
            layer_wise (bool): Group parameters by layer rather than fixed size
            group_size (int, optional): Fixed number of parameters per group when layer_wise=False
            eps (float): Small constant for numerical stability
            scale_aware (bool): Preserve some gradient scale information
            scale_factor (float): Mix factor for scale-aware normalization
            max_group_size (int, optional): Maximum parameters per sub-group (if applicable, None for no limit)
            clip_norm (float, optional): Clip gradient norm value
            adaptive (bool): Use adaptive gradient scaling like RMSprop
            adaptive_eps (float): Small constant for adaptive scaling
            disable_layer_mapping (bool): Disable layer mapping for normalization
            profile_time (bool): Enable profiling of normalization and SGD steps
            use_cached_mapping (bool): Create and save layer mapping cache at initialization
            layer_lr_multipliers (dict, optional): Dictionary mapping layer indices to LR multipliers. Effective when layer_wise=True.
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
            max_group_size=max_group_size,  # Updated default
            clip_norm=clip_norm,
            adaptive=adaptive,
            adaptive_eps=adaptive_eps,
            disable_layer_mapping=disable_layer_mapping,
            profile_time=profile_time,
            use_cached_mapping=use_cached_mapping,
            layer_lr_multipliers=layer_lr_multipliers if layer_lr_multipliers is not None else {}, # Added
            # CUDA settings
            use_cuda_kernels=use_cuda_kernels,
            force_cuda_fallback=force_cuda_fallback,
            # Performance tuning
            normalize_interval=normalize_interval,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

        # Create parameter name mapping for layer-wise grouping
        self.param_to_layer = {}
        self._cached_buffers = {}  # Cache for pre-allocated buffers
        self.profile_stats = {"normalize_time": 0.0, "sgd_time": 0.0, "cuda_time": 0.0, "total_steps": 0}
        self._cuda_available = CUDA_OPS_AVAILABLE and not force_cuda_fallback
        
        # If layer_wise is enabled, organize parameters by layer
        # Always create mapping if needed, save if use_cached_mapping is True
        if layer_wise and normalize and not disable_layer_mapping:
            print("Organizing layer groups...")
            self._organize_layer_groups()
            if use_cached_mapping:
                try:
                    torch.save(self.param_to_layer, "milo_layer_mapping.pt")
                    print("Saved layer mapping to milo_layer_mapping.pt")
                except Exception as e:
                    print(f"Warning: Could not save layer mapping cache: {e}")
                    
        # Initialize CUDA optimization metadata if available
        if self._cuda_available:
            self._init_cuda_metadata()

    def _organize_layer_groups(self):
        """
        Organize parameters into layer groups based on parameter names.
        
        This enables layer-wise normalization where gradients from the same
        logical layer are normalized together, preserving intra-layer relationships.
        """
        # Build unique layer names from parameter groups
        self.param_to_layer = {}
        
        # Try to extract parameter names from their full names
        for group_idx, group in enumerate(self.param_groups):
            for param_idx, param in enumerate(group['params']):
                # Use param identifier with group/param index as backup names
                param_id = f"group{group_idx}_param{param_idx}"
                
                # Try to find parameter name in state_dict keys
                for name, p in self._params_to_names(param, param_id).items():
                    if p is param:
                        # Extract the layer name (e.g., 'conv1.weight' -> 'conv1')
                        layer_name = name.split('.')[0] if '.' in name else name
                        self.param_to_layer[param] = layer_name
        
        # Create mapping from layer name to index
        layer_names = sorted(set(self.param_to_layer.values()))
        layer_indices = {name: idx for idx, name in enumerate(layer_names)}
        
        # Map each parameter to its layer index
        for param, layer_name in self.param_to_layer.items():
            self.param_to_layer[param] = layer_indices.get(layer_name, 0)
    
    def _init_cuda_metadata(self):
        """Initialize metadata needed for CUDA kernels."""
        self._cuda_metadata = {}
        
        for group in self.param_groups:
            if group["layer_wise"]:
                layer_params = {}
                for param in group["params"]:
                    layer_idx = self.param_to_layer.get(param, 0)
                    if layer_idx not in layer_params:
                        layer_params[layer_idx] = []
                    layer_params[layer_idx].append(param)
                
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
    
    def _params_to_names(self, target_param, default_name):
        """Helper to find parameter names for layer-wise grouping."""
        # This is a simplified version that uses a default name scheme
        # In practice, you would scan the model's state_dict to find actual names
        return {default_name: target_param}

    def __setstate__(self, state):
        """Restore the optimizer state.

        Args:
            state (dict): The state dictionary to load.
        """
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("differentiable", False)
            # Normalization defaults
            group.setdefault("normalize", True)
            group.setdefault("layer_wise", True)
            group.setdefault("group_size", None)
            group.setdefault("eps", 1e-5)
            group.setdefault("scale_aware", False)
            group.setdefault("scale_factor", 0.2)
            group.setdefault("max_group_size", None)  # Updated default
            group.setdefault("clip_norm", None)
            group.setdefault("adaptive", False)
            group.setdefault("adaptive_eps", 1e-8)
            group.setdefault("disable_layer_mapping", False)
            group.setdefault("profile_time", False)
            group.setdefault("use_cached_mapping", False)
            group.setdefault("layer_lr_multipliers", {}) # Added
            group.setdefault("normalize_interval", 1)

    def _init_group(self, group, params, grads, momentum_buffer_list):
        """Initialize a parameter group for the optimization step.

        This method populates lists of parameters, gradients, and momentum buffers
        for the current group. It also handles gradient clipping and initializes
        adaptive scaling states if required.

        Args:
            group (dict): The parameter group being processed.
            params (list): An empty list to be filled with parameter tensors.
            grads (list): An empty list to be filled with gradient tensors.
            momentum_buffer_list (list): An empty list to be filled with momentum buffers.

        Returns:
            bool: True if any gradient in the group is sparse, False otherwise.
        """
        has_sparse_grad = False

        for p in group["params"]:
            if p.grad is not None:
                # Apply gradient clipping if specified
                if group["clip_norm"] is not None:
                    torch.nn.utils.clip_grad_norm_(p, group["clip_norm"])
                    
                params.append(p)
                grads.append(p.grad)
                if p.grad.is_sparse:
                    has_sparse_grad = True

                if group["momentum"] != 0:
                    state = self.state[p]
                    momentum_buffer_list.append(state.get("momentum_buffer"))

                # Setup adaptive scaling state if needed
                if group["adaptive"] and "sum_sq_grad" not in self.state[p]:
                    self.state[p]["sum_sq_grad"] = torch.zeros_like(p.data)

        return has_sparse_grad

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        profile_time = any(group.get('profile_time', False) for group in self.param_groups)
        total_start = time.time() if profile_time else None
        
        for group in self.param_groups:
            params = []
            grads = []
            momentum_buffer_list = []

            # Initialize and collect parameters with gradients
            has_sparse_grad = self._init_group(group, params, grads, momentum_buffer_list)

            # Apply gradient normalization if enabled
            if group["normalize"]:
                group["_norm_step"] = group.get("_norm_step", 0) + 1
                do_norm = (group.get("normalize_interval", 1) <= 1) or (
                    group["_norm_step"] % group.get("normalize_interval", 1) == 0
                )
                if do_norm:
                    norm_start = time.time() if profile_time else None
                    self._normalize_gradients(group, params, grads)
                    if profile_time and norm_start:
                        self.profile_stats["normalize_time"] += time.time() - norm_start

            # Choose update routine based on configuration
            sgd_start = time.time() if profile_time else None
            if group["foreach"] and not torch.jit.is_scripting():
                func = self._multi_tensor_normalized
            else:
                func = self._single_tensor_normalized

            func(
                params,
                grads,
                momentum_buffer_list,
                group, # Pass group
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                lr=group["lr"],
                dampening=group["dampening"],
                nesterov=group["nesterov"],
                maximize=group["maximize"],
                has_sparse_grad=has_sparse_grad,
                adaptive=group["adaptive"],
                adaptive_eps=group["adaptive_eps"],
            )

            if profile_time and sgd_start:
                self.profile_stats["sgd_time"] += time.time() - sgd_start

            # Update momentum buffers in state
            if group["momentum"] != 0:
                for p, momentum_buffer in zip(params, momentum_buffer_list):
                    state = self.state[p]
                    state["momentum_buffer"] = momentum_buffer

        if profile_time and total_start:
            self.profile_stats["total_steps"] += 1
            if self.profile_stats["total_steps"] % 100 == 0:
                cuda_status = "CUDA" if self._cuda_available else "PyTorch"
                print(f"MILO Profiling ({cuda_status}) - Normalization: {self.profile_stats['normalize_time']/self.profile_stats['total_steps']*1000:.2f}ms/step, "
                      f"SGD: {self.profile_stats['sgd_time']/self.profile_stats['total_steps']*1000:.2f}ms/step")

        return loss
    
    def _normalize_gradients(self, group, params, grads):
        """
        Apply normalization to gradients based on group settings with enhanced CUDA acceleration.
        """
        if not group["normalize"] or not grads:
            return
            
        # Enhanced CUDA acceleration with fused operations
        use_cuda = (self._cuda_available and 
                   group["use_cuda_kernels"] and 
                   not group.get("force_cuda_fallback", False) and
                   all(g.is_cuda for g in grads))

        if use_cuda:
            try:
                # Use fused CUDA operations for better performance
                from milo_cuda_ops import milo_cuda
                
                if group["layer_wise"] and self.param_to_layer:
                    # Layer-wise: ALWAYS use group-based normalization within the layer
                    meta = getattr(self, "_cuda_metadata", {}).get(id(group)) if hasattr(self, "_cuda_metadata") else None
                    if meta and 'layer_params' in meta:
                        layer_map = meta['layer_params']
                    else:
                        # Build on the fly
                        layer_map = {}
                        for p in params:
                            layer_idx = self.param_to_layer.get(p, 0)
                            layer_map.setdefault(layer_idx, []).append(p)

                    for layer_idx, plist in layer_map.items():
                        layer_params = []
                        layer_grads = []
                        for p in plist:
                            if p.grad is not None:
                                layer_params.append(p)
                                layer_grads.append(p.grad)
                        if layer_grads:
                            # Use fixed-size group normalization (auto group size if None)
                            self._normalize_fixed_size_groups_batch(group, layer_params, layer_grads)
                else:
                    # Non–layer-wise path: use fixed-size group normalization
                    # Auto-determines group size when group_size is None
                    self._normalize_fixed_size_groups_batch(group, params, grads)
                return
            except Exception as e:
                if group.get("profile_time", False):
                    print(f"Warning: CUDA normalization failed ({e}), falling back to PyTorch")
        
        # Optimized PyTorch fallback with better memory efficiency
        layer_wise = group["layer_wise"]
        disable_layer_mapping = group.get("disable_layer_mapping", False)
        
        if disable_layer_mapping or (layer_wise and not self.param_to_layer):
            # Use vectorized operations for parameter-wise normalization
            for grad in grads:
                self._normalize_single_grad_optimized(grad, group)
            return
            
        if layer_wise:
            # Group-based normalization within each layer (auto group size if None)
            meta = getattr(self, "_cuda_metadata", {}).get(id(group)) if hasattr(self, "_cuda_metadata") else None
            if meta and 'layer_params' in meta:
                for _, plist in meta['layer_params'].items():
                    layer_params = []
                    layer_grads = []
                    for p in plist:
                        if p.grad is not None:
                            layer_params.append(p)
                            layer_grads.append(p.grad)
                    if layer_grads:
                        self._normalize_fixed_size_groups_batch(group, layer_params, layer_grads)
            else:
                # Build on the fly using current params/grads
                layer_map = {}
                for p, g in zip(params, grads):
                    idx = self.param_to_layer.get(p, 0)
                    layer_map.setdefault(idx, []).append((p, g))
                for _, pairs in layer_map.items():
                    layer_params = [p for p, g in pairs]
                    layer_grads = [g for p, g in pairs]
                    self._normalize_fixed_size_groups_batch(group, layer_params, layer_grads)
        else:
            # Optimized fixed-size group normalization across each tensor
            # Auto group sizing when group_size is None
            self._normalize_fixed_size_groups_batch(group, params, grads)

    def _normalize_single_grad_optimized(self, grad, group):
        """Optimized single gradient normalization with CUDA-friendly operations."""
        eps = group["eps"]
        scale_aware = group["scale_aware"]
        scale_factor = group["scale_factor"]
        
        # Use more efficient norm computation
        if grad.is_cuda:
            # GPU-optimized path
            grad_norm = torch.linalg.vector_norm(grad)
        else:
            # CPU path
            grad_norm = torch.norm(grad)
        
        if grad_norm > eps:
            normalization_factor = 1.0 / grad_norm
            
            if scale_aware:
                clamped_norm = grad_norm.clamp(max=1.0)
                # In-place operations for better GPU efficiency
                if grad.is_cuda:
                    # Avoid host sync by keeping computation on device
                    normalized_term = (grad * normalization_factor) * clamped_norm
                    grad.mul_(scale_factor)
                    grad.add_(normalized_term, alpha=(1 - scale_factor))
                else:
                    normalized_grad = grad * normalization_factor
                    grad.copy_(grad * scale_factor + normalized_grad * (1 - scale_factor) * clamped_norm)
            else:
                # Simple in-place normalization
                grad.mul_(normalization_factor)

    def _normalize_layer_grads_optimized(self, layer_grads, group):
        """Optimized layer-wise gradient normalization with minimal memory allocation."""
        eps = group["eps"]
        scale_aware = group["scale_aware"]
        scale_factor = group["scale_factor"]
        
        # Compute norm in a two-pass fashion to avoid concatenation and extra allocations
        # Pass 1: accumulate squared norm
        device = layer_grads[0].device
        dtype = layer_grads[0].dtype
        sq_sum = torch.zeros(1, device=device, dtype=torch.float32)
        for g in layer_grads:
            # compute in float32 for numerical stability even if params are float16/bfloat16
            sq_sum.add_(g.detach().to(torch.float32).pow(2).sum())
        grad_norm = sq_sum.sqrt()
        
        if grad_norm > eps:
            normalization_factor = 1.0 / grad_norm
            
            # Apply normalization to each gradient in-place
            for grad in layer_grads:
                if scale_aware:
                    clamped_norm = grad_norm.clamp(max=1.0)
                    if grad.is_cuda:
                        # GPU-optimized in-place operations
                        grad.mul_(scale_factor).add_(grad * normalization_factor, 
                                  alpha=(1 - scale_factor) * float(clamped_norm))
                    else:
                        # CPU fallback
                        normalized_grad = grad * normalization_factor
                        grad.copy_(grad * scale_factor + normalized_grad * (1 - scale_factor) * float(clamped_norm))
                else:
                    grad.mul_(normalization_factor)

    def _normalize_layer_with_buffer(self, group, params, grads, sizes, buffer):
        """Optimized version of normalize_layer_optimized that uses a pre-allocated buffer."""
        eps = group["eps"]
        scale_aware = group["scale_aware"]
        scale_factor = group["scale_factor"]
        
        # Copy gradients into buffer without creating intermediate tensors
        start_idx = 0
        for i, grad in enumerate(grads):
            grad_size = sizes[i]
            buffer[start_idx:start_idx + grad_size].copy_(grad.view(-1))
            start_idx += grad_size
        
        # Check for numerical instability in this layer's gradients
        buffer_slice = buffer[:start_idx]
        max_val = buffer_slice.abs().max().item()
        
        # If gradients are extremely large or small, use a more robust normalization approach
        if max_val > 1e4 or max_val < 1e-6:
            # Clip extreme values to prevent instability
            buffer_slice.clamp_(-1e4, 1e4)
            # Use a more robust norm that's less sensitive to outliers
            robust_mean = torch.median(buffer_slice) # Changed from .mean() to torch.median()
            # Use median absolute deviation instead of standard deviation for outlier resilience
            abs_dev = (buffer_slice - robust_mean).abs()
            robust_scale = abs_dev.median() * 1.4826 + eps  # Scale factor approximates std for normal distribution
            
            # Apply robust normalization
            buffer_slice.sub_(robust_mean).div_(robust_scale)
        else:
            # Standard normalization path for stable cases
            layer_mean = torch.mean(buffer_slice)
            layer_std = torch.std(buffer_slice) + eps
            
            # Apply normalization in-place with vectorized operations
            if scale_aware:
                # Store original values for scale-aware normalization
                if scale_factor < 1.0 and scale_factor > 0:
                    orig_vals = buffer_slice.clone()
                    # Normalize
                    buffer_slice.sub_(layer_mean).div_(layer_std)
                    # Mix with original values for scale-aware normalization
                    buffer_slice.mul_(1 - scale_factor).add_(orig_vals.mul(scale_factor))
            else:
                # Standard normalization (faster without scale-aware)
                buffer_slice.sub_(layer_mean).div_(layer_std)
        
        # Copy normalized gradients back
        start_idx = 0
        for i, grad in enumerate(grads):
            grad_size = sizes[i]
            grad.copy_(buffer[start_idx:start_idx + grad_size].view_as(grad))
            start_idx += grad_size

    def _normalize_fixed_size_groups_batch(self, group, params, grads):
        """
        Batch process fixed-size groups for normalization.
        """
        eps = group["eps"]
        given_group_size = group["group_size"]
        scale_aware = group["scale_aware"]
        scale_factor = group["scale_factor"]

        for param, grad in zip(params, grads):
            flat_grad = grad.view(-1)
            N = flat_grad.numel()

            if given_group_size is not None:
                group_size = given_group_size
            else:
                dynamic_num_groups = max(1, int(math.sqrt(N)))
                group_size = math.ceil(N / dynamic_num_groups)

            if group_size < 2 or N < 2:
                continue

            remainder = N % group_size
            if remainder != 0:
                pad_size = group_size - remainder
                flat_grad_padded = torch.cat([flat_grad, flat_grad.new_zeros(pad_size)], dim=0)
            else:
                flat_grad_padded = flat_grad

            reshaped = flat_grad_padded.view(-1, group_size)
            group_mean = reshaped.mean(dim=1, keepdim=True)
            group_std = reshaped.std(dim=1, keepdim=True) + eps

            if scale_aware:
                normalized = scale_factor * reshaped + (1 - scale_factor) * ((reshaped - group_mean) / group_std)
            else:
                normalized = (reshaped - group_mean) / group_std

            normalized_flat_grad = normalized.view(-1)[:N]
            grad.copy_(normalized_flat_grad.view_as(grad))

    def _single_tensor_normalized(
        self,
        params: List[Tensor],
        grads: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        group: dict, # Add group parameter
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        maximize: bool,
        has_sparse_grad: bool,
        adaptive: bool,
        adaptive_eps: float,
    ):
        """Performs optimized SGD update for single tensors with enhanced CUDA support.

        This method includes CUDA-accelerated operations and memory-efficient updates
        for better GPU performance.
        """
        # Try CUDA-accelerated batch updates if possible
        use_cuda = (self._cuda_available and 
                   all(p.is_cuda for p in params) and
                   group["use_cuda_kernels"] and 
                   not group.get("force_cuda_fallback", False))

        if use_cuda and len(params) > 1:
            try:
                # Use CUDA acceleration for batch operations
                from milo_cuda_ops import milo_cuda
                self._cuda_batch_update(params, grads, momentum_buffer_list, group,
                                      weight_decay, momentum, lr, dampening, 
                                      nesterov, maximize, adaptive, adaptive_eps)
                return
            except Exception as e:
                if group.get("profile_time", False):
                    print(f"Warning: CUDA batch update failed ({e}), falling back")

        # Optimized per-parameter updates
        for i, param in enumerate(params):
            grad = grads[i] if not maximize else -grads[i]

            # Weight decay with optimized operations
            if weight_decay != 0:
                if param.is_cuda:
                    # In-place GPU operation
                    grad = grad.add(param, alpha=weight_decay)
                else:
                    grad = grad.add(param, alpha=weight_decay)

            # Optimized momentum handling
            if momentum != 0:
                buf = momentum_buffer_list[i]

                if buf is None:
                    buf = torch.clone(grad).detach()
                    momentum_buffer_list[i] = buf
                else:
                    if param.is_cuda:
                        # GPU-optimized in-place momentum update
                        buf.mul_(momentum).add_(grad, alpha=1 - dampening)
                    else:
                        buf.mul_(momentum).add_(grad, alpha=1 - dampening)

                if nesterov:
                    grad = grad.add(buf, alpha=momentum)
                else:
                    grad = buf
                    
            # Enhanced adaptive scaling with GPU optimization
            if adaptive:
                state = self.state[params[i]]
                if 'sum_sq_grad' not in state:
                    state['sum_sq_grad'] = grad.detach().pow(2)
                else:
                    if param.is_cuda:
                        # GPU-optimized adaptive update
                        state['sum_sq_grad'].add_(grad.detach().pow(2))
                    else:
                        state['sum_sq_grad'].add_(grad.detach().pow(2))
                    
                # Optimized division for GPU
                adaptive_scale = state['sum_sq_grad'].sqrt().add_(adaptive_eps)
                grad = grad.div(adaptive_scale)

            # Learning rate adjustment with layer multipliers
            current_lr = lr
            if group.get('layer_lr_multipliers'):
                layer_idx = self.param_to_layer.get(param)
                if layer_idx is not None:
                    multiplier = group['layer_lr_multipliers'].get(layer_idx, 1.0)
                    current_lr *= multiplier
            
            # Optimized parameter update
            if param.is_cuda:
                # GPU-optimized in-place update
                param.data.add_(grad, alpha=-current_lr)
            else:
                param.data.add_(grad, alpha=-current_lr)

    def _cuda_batch_update(self, params, grads, momentum_buffer_list, group,
                          weight_decay, momentum, lr, dampening, nesterov, 
                          maximize, adaptive, adaptive_eps):
        """CUDA-accelerated batch parameter updates."""
        from milo_cuda_ops import milo_cuda
        
        # Prepare batch data
        param_data = [p.data for p in params]
        grad_data = [-g if maximize else g for g in grads]
        
        # Apply weight decay if needed
        if weight_decay != 0:
            for i, (param, grad) in enumerate(zip(param_data, grad_data)):
                grad_data[i] = grad.add(param, alpha=weight_decay)
        
        # Handle momentum in batch
        if momentum != 0:
            momentum_data = []
            for i, buf in enumerate(momentum_buffer_list):
                if buf is None:
                    buf = torch.clone(grad_data[i]).detach()
                    momentum_buffer_list[i] = buf
                else:
                    # Use CUDA-accelerated momentum update
                    milo_cuda.enhanced_momentum_update(
                        buf, buf, grad_data[i], momentum, 1-dampening
                    )
                momentum_data.append(buf)
            
            if nesterov:
                for i in range(len(grad_data)):
                    grad_data[i] = grad_data[i].add(momentum_data[i], alpha=momentum)
            else:
                grad_data = momentum_data
        
        # Apply adaptive scaling if needed
        if adaptive:
            for i, param in enumerate(params):
                state = self.state[param]
                if 'sum_sq_grad' not in state:
                    state['sum_sq_grad'] = grad_data[i].detach().pow(2)
                else:
                    state['sum_sq_grad'].add_(grad_data[i].detach().pow(2))
                
                adaptive_scale = state['sum_sq_grad'].sqrt().add_(adaptive_eps)
                grad_data[i] = grad_data[i].div(adaptive_scale)
        
        # Apply learning rates and update parameters
        learning_rates = [lr] * len(params)
        if group.get('layer_lr_multipliers'):
            for i, param in enumerate(params):
                layer_idx = self.param_to_layer.get(param)
                if layer_idx is not None:
                    multiplier = group['layer_lr_multipliers'].get(layer_idx, 1.0)
                    learning_rates[i] *= multiplier
        
        # Fast path: use foreach when all params are on same device/dtype and a single LR
        can_use_foreach = (
            not group.get('layer_lr_multipliers') and
            isinstance(lr, (float, int)) and
            len(param_data) > 1 and
            all(p.is_cuda for p in param_data) and
            len({(p.device, p.dtype) for p in param_data}) == 1 and
            len({g.dtype for g in grad_data}) == 1
        )

        if can_use_foreach:
            scaled = torch._foreach_mul(grad_data, -float(lr))
            torch._foreach_add_(param_data, scaled)
        else:
            for param, grad, current_lr in zip(param_data, grad_data, learning_rates):
                param.add_(grad, alpha=-current_lr)

    def _multi_tensor_normalized(
        self,
        params: List[Tensor],
        grads: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        group: dict, # Add group parameter
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        maximize: bool,
        has_sparse_grad: bool,
        adaptive: bool,
        adaptive_eps: float,
    ):
        """Performs the SGD update for multiple tensors using foreach operations.

        This method is used when the `foreach` option is True. It leverages PyTorch's
        `_foreach_*` operations for potentially faster updates on multiple tensors
        simultaneously. It applies weight decay, momentum, Nesterov momentum, adaptive
        scaling (if enabled), and learning rate updates.

        Args:
            params (List[Tensor]): List of parameters in the current group.
            grads (List[Tensor]): List of corresponding gradients.
            momentum_buffer_list (List[Optional[Tensor]]): List of momentum buffers.
            group (dict): The parameter group dictionary, containing optimizer settings like layer_lr_multipliers.
            weight_decay (float): Weight decay factor.
            momentum (float): Momentum factor.
            lr (float): Learning rate.
            dampening (float): Dampening factor for momentum.
            nesterov (bool): Whether to use Nesterov momentum.
            maximize (bool): Whether to maximize (negate gradients).
            has_sparse_grad (bool): Indicates if sparse gradients are present.
            adaptive (bool): Whether to use adaptive gradient scaling.
            adaptive_eps (float): Epsilon for adaptive scaling stability.
        """
        if len(params) == 0:
            return

        # Group tensors by device and dtype
        grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
            [params, grads, momentum_buffer_list], with_indices=True  # type: ignore[list-item]
        )

        for (
            device_params_,
            device_grads_,
            device_momentum_buffer_list,
        ), indices in grouped_tensors.values():
            device_params: List[Tensor] = cast(List[Tensor], device_params_)
            device_grads: List[Tensor] = cast(List[Tensor], device_grads_)

            device_has_sparse_grad = has_sparse_grad and any(
                grad.is_sparse for grad in device_grads
            )

            if maximize:
                device_grads = torch._foreach_neg(device_grads)  # type: ignore[assignment]

            if weight_decay != 0:
                # Re-use the intermediate memory (device_grads) already allocated for maximize
                if maximize:
                    torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
                else:
                    device_grads = torch._foreach_add(  # type: ignore[assignment]
                        device_grads, device_params, alpha=weight_decay
                    )

            if momentum != 0:
                bufs: List[Tensor] = []

                all_states_with_momentum_buffer = True
                for i in range(len(device_momentum_buffer_list)):
                    if device_momentum_buffer_list[i] is None:
                        all_states_with_momentum_buffer = False
                        break
                    else:
                        bufs.append(cast(Tensor, device_momentum_buffer_list[i]))

                if all_states_with_momentum_buffer:
                    torch._foreach_mul_(bufs, momentum)
                    torch._foreach_add_(bufs, device_grads, alpha=1 - dampening)
                else:
                    bufs = []
                    for i in range(len(device_momentum_buffer_list)):
                        if device_momentum_buffer_list[i] is None:
                            buf = device_momentum_buffer_list[i] = momentum_buffer_list[
                                indices[i]
                            ] = torch.clone(device_grads[i]).detach()
                        else:
                            buf = cast(Tensor, device_momentum_buffer_list[i])
                            buf.mul_(momentum).add_(device_grads[i], alpha=1 - dampening)

                        bufs.append(buf)

                if nesterov:
                    torch._foreach_add_(device_grads, bufs, alpha=momentum)
                else:
                    device_grads = bufs
                    
            # Apply adaptive scaling if enabled
            if adaptive:
                for i, param in enumerate(device_params):
                    state = self.state[param]
                    if 'sum_sq_grad' not in state:
                        state['sum_sq_grad'] = device_grads[i].detach().pow(2)
                    else:
                        state['sum_sq_grad'].add_(device_grads[i].detach().pow(2))
                        
                    # Scale by inverse sqrt of sum - we have to do this individually
                    # since this isn't supported by foreach ops yet
                    device_grads[i] = device_grads[i] / (state['sum_sq_grad'].sqrt() + adaptive_eps)

            # Determine if foreach can be used for the final step
            can_use_foreach_for_final_step = (
                not device_has_sparse_grad and
                not group.get('layer_lr_multipliers') and
                not (isinstance(lr, torch.Tensor) and torch.compiler.is_compiling())
            )
            can_use_foreach_for_final_step_tensor_lr_compiled = (
                not device_has_sparse_grad and
                not group.get('layer_lr_multipliers') and
                isinstance(lr, torch.Tensor) and torch.compiler.is_compiling()
            )

            if can_use_foreach_for_final_step:
                # Original foreach for non-tensor lr
                torch._foreach_add_(device_params, device_grads, alpha=-lr)
            elif can_use_foreach_for_final_step_tensor_lr_compiled:
                # Original foreach for tensor lr when compiling
                grads_x_lr = torch._foreach_mul(device_grads, -lr)
                # Use _foreach_add to data attribute
                for i, p in enumerate(device_params):
                    p.data.add_(grads_x_lr[i])
            else:
                # Fallback to loop if sparse gradients, layer_lr_multipliers, or specific tensor lr case
                for i, p_device in enumerate(device_params):
                    current_grad_val = device_grads[i]
                    
                    # Determine the learning rate for this parameter
                    current_lr_val = lr.item() if isinstance(lr, torch.Tensor) else lr

                    if group.get('layer_lr_multipliers'):
                        # p_device is the parameter tensor, self.param_to_layer maps param tensors to layer indices
                        layer_idx = self.param_to_layer.get(p_device)
                        if layer_idx is not None: # layer_idx might be None if layer_wise is False or param not in map
                            multiplier = group['layer_lr_multipliers'].get(layer_idx, 1.0)
                            current_lr_val *= multiplier
                    
                    p_device.data.add_(current_grad_val, alpha=-current_lr_val)
