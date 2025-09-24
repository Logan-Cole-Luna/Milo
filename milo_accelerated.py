"""
Accelerated implementation of the Milo optimizer.

This file defines a subclass of PyTorch's :class:`Optimizer` that
implements Normalized Stochastic Gradient Descent with several
performance improvements:

* Group sizes for gradient normalization are computed once per
  parameter and cached in the parameter's state dictionary.
* Padding buffers used to align gradient vectors to group size are
  cached and reused rather than reallocated each step.
* The rest of the optimizer logic remains unchanged from the
  reference implementation, preserving the per‑parameter grouping
  semantics (layer‑wise or fixed‑size groups).

This class can be used as a drop‑in replacement for the original
``milo`` optimizer.  Simply import ``milo_accelerated.milo`` instead
of the original implementation.
"""

import math
import time
from typing import List, Optional, Union, Dict

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT

CUDA_OPS_AVAILABLE = False  # CUDA kernel support is not available in this example

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
        use_cuda_kernels: bool = True,
        force_cuda_fallback: bool = False,
        normalize_interval: int = 1,
        verbose_profile: bool = False,
    ):
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
            layer_lr_multipliers=layer_lr_multipliers if layer_lr_multipliers is not None else {},
            use_cuda_kernels=use_cuda_kernels,
            force_cuda_fallback=force_cuda_fallback,
            normalize_interval=normalize_interval,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

        # If verbose_profile is set, more detailed timing information
        # will be printed at each optimization step.  This should only
        # be enabled for debugging purposes, as it can noticeably slow
        # down training.
        self.verbose_profile = verbose_profile

        self.param_to_layer: Dict[Tensor, int] = {}
        self._cached_buffers: Dict[str, Tensor] = {}
        self.profile_stats = {"normalize_time": 0.0, "sgd_time": 0.0, "cuda_time": 0.0, "total_steps": 0}
        self._cuda_available = CUDA_OPS_AVAILABLE and not force_cuda_fallback

        # Initialize layer mapping if necessary
        if layer_wise and normalize and not disable_layer_mapping:
            self._organize_layer_groups()
        if self._cuda_available:
            self._init_cuda_metadata()

    def _organize_layer_groups(self) -> None:
        self.param_to_layer = {}
        for group_idx, group in enumerate(self.param_groups):
            for param_idx, param in enumerate(group['params']):
                param_id = f"group{group_idx}_param{param_idx}"
                # In a full implementation, we would map real parameter names; here we use indices
                layer_name = param_id.split('.')[0] if '.' in param_id else param_id
                self.param_to_layer[param] = layer_name
        # Assign numeric indices
        layer_names = sorted(set(self.param_to_layer.values()))
        layer_indices = {name: idx for idx, name in enumerate(layer_names)}
        for param, layer_name in self.param_to_layer.items():
            self.param_to_layer[param] = layer_indices[layer_name]

    def _init_cuda_metadata(self) -> None:
        # This example does not support CUDA kernels; metadata is left empty
        self._cuda_metadata = {}

    def _init_group(self, group, params, grads, momentum_buffer_list) -> bool:
        has_sparse_grad = False
        for p in group["params"]:
            if p.grad is not None:
                if group["clip_norm"] is not None:
                    torch.nn.utils.clip_grad_norm_(p, group["clip_norm"])
                params.append(p)
                grads.append(p.grad)
                if p.grad.is_sparse:
                    has_sparse_grad = True
                if group["momentum"] != 0:
                    state = self.state[p]
                    momentum_buffer_list.append(state.get("momentum_buffer"))
                if group["adaptive"] and "sum_sq_grad" not in self.state[p]:
                    self.state[p]["sum_sq_grad"] = torch.zeros_like(p.data)
        return has_sparse_grad

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            # Time individual stages when verbose profiling is enabled
            t_init_start: Optional[float] = None
            t_norm_start: Optional[float] = None
            t_norm: float = 0.0
            if self.verbose_profile:
                t_init_start = time.perf_counter()
            params: List[Tensor] = []
            grads: List[Tensor] = []
            momentum_buffers: List[Optional[Tensor]] = []
            has_sparse = self._init_group(group, params, grads, momentum_buffers)
            if self.verbose_profile and t_init_start is not None:
                t_init = time.perf_counter() - t_init_start
            if group["normalize"]:
                group["_norm_step"] = group.get("_norm_step", 0) + 1
                if group["_norm_step"] % group.get("normalize_interval", 1) == 0:
                    if self.verbose_profile:
                        t_norm_start = time.perf_counter()
                    self._normalize_gradients(group, params, grads)
                    if self.verbose_profile and t_norm_start is not None:
                        t_norm = time.perf_counter() - t_norm_start
            # Update parameters
            if self.verbose_profile:
                t_update_start = time.perf_counter()
            self._single_tensor_normalized(
                params,
                grads,
                momentum_buffers,
                group,
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                lr=group["lr"],
                dampening=group["dampening"],
                nesterov=group["nesterov"],
                maximize=group["maximize"],
                has_sparse_grad=has_sparse,
                adaptive=group["adaptive"],
                adaptive_eps=group["adaptive_eps"],
            )
            if self.verbose_profile:
                t_update = time.perf_counter() - t_update_start
                # Print profiling for this group
                msg = f"[Profiling] Group finished: init={t_init:.6f}s"
                # If normalization was applied, t_norm will be > 0
                if t_norm > 0:
                    msg += f", normalize={t_norm:.6f}s"
                msg += f", update={t_update:.6f}s"
                print(msg)
            if group["momentum"] != 0:
                for p, buf in zip(params, momentum_buffers):
                    self.state[p]["momentum_buffer"] = buf
        return loss

    def _normalize_gradients(self, group, params, grads) -> None:
        if not group["normalize"] or not grads:
            return
        # Only CPU path is implemented in this example
        if group["layer_wise"]:
            layer_map: Dict[int, List[int]] = {}
            for i, p in enumerate(params):
                layer_idx = self.param_to_layer.get(p, 0)
                layer_map.setdefault(layer_idx, []).append(i)
            for _, indices in layer_map.items():
                layer_params = [params[i] for i in indices]
                layer_grads = [grads[i] for i in indices]
                self._normalize_fixed_size_groups_batch(group, layer_params, layer_grads)
        else:
            self._normalize_fixed_size_groups_batch(group, params, grads)

    def _normalize_fixed_size_groups_batch(self, group, params, grads) -> None:
        """Normalize gradients using fixed-size or dynamically computed groups.

        This accelerated version caches the computed group size and zero padding
        buffer in the state of each parameter, avoiding redundant calls to
        ``math.sqrt`` and repeated tensor allocations.  The grouping logic
        (sqrt of the number of elements) and scale-aware mixing remain
        unchanged.
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
                # Use cached group size if available
                state = self.state[param]
                group_size = state.get("_norm_group_size")
                if group_size is None:
                    dynamic_num_groups = max(1, int(math.sqrt(N)))
                    group_size = math.ceil(N / dynamic_num_groups)
                    state["_norm_group_size"] = group_size
            if group_size < 2 or N < 2:
                continue
            remainder = N % group_size
            if remainder != 0:
                pad_size = group_size - remainder
                # Reuse or allocate padding buffer in state
                state = self.state[param]
                pad = state.get("_norm_pad_zeros")
                if pad is None or pad.numel() < pad_size:
                    pad = torch.zeros(pad_size, dtype=grad.dtype, device=grad.device)
                    state["_norm_pad_zeros"] = pad
                flat_padded = torch.cat([flat_grad, pad[:pad_size]])
            else:
                flat_padded = flat_grad
            reshaped = flat_padded.view(-1, group_size)
            group_mean = reshaped.mean(dim=1, keepdim=True)
            group_std = reshaped.std(dim=1, keepdim=True) + eps
            if scale_aware:
                normalized = scale_factor * reshaped + (1.0 - scale_factor) * ((reshaped - group_mean) / group_std)
            else:
                normalized = (reshaped - group_mean) / group_std
            normalized_flat = normalized.view(-1)[:N]
            grad.copy_(normalized_flat.view_as(grad))

    def _single_tensor_normalized(
        self,
        params: List[Tensor],
        grads: List[Tensor],
        momentum_buffers: List[Optional[Tensor]],
        group: dict,
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
    ) -> None:
        """
        Perform the SGD update on a list of tensors.  This implementation
        attempts to minimise Python overhead by using foreach operations
        for weight decay and momentum updates when possible.  It also
        measures the time spent in each logical sub‑component (weight
        decay, momentum update, adaptive scaling and parameter update)
        when ``self.verbose_profile`` is enabled.  These timings can
        help diagnose bottlenecks in the update phase.
        """
        # Convert negative update if maximize is requested
        if maximize:
            # Invert all gradients in place.  We make a shallow copy
            # because modifying ``grads`` in place may affect upstream
            # callers who expect the original gradients.
            for i in range(len(grads)):
                grads[i] = -grads[i]
        # Profiling accumulators
        t_weight_decay = 0.0
        t_momentum = 0.0
        t_adaptive = 0.0
        t_final_update = 0.0
        # Apply weight decay in a batched manner if possible
        if weight_decay != 0:
            if self.verbose_profile:
                t0 = time.perf_counter()
            try:
                # Use foreach to add weight decay: grads += weight_decay * params
                # We extract the parameter data tensors for addition
                param_data = [p.data for p in params]
                # ``torch._foreach_add`` returns a new list of tensors; to keep
                # ``grads`` consistent, replace its elements with the returned
                # tensors.  This avoids allocating new Python lists while
                # preserving references to the original gradient tensors.
                updated_grads = torch._foreach_add(grads, param_data, alpha=weight_decay)
                for i in range(len(grads)):
                    grads[i] = updated_grads[i]
            except Exception:
                # Fall back to scalar loop if foreach fails (e.g. due to
                # heterogeneous dtypes or devices)
                for i, (g, p) in enumerate(zip(grads, params)):
                    grads[i] = g.add(p, alpha=weight_decay)
            if self.verbose_profile:
                t_weight_decay += time.perf_counter() - t0
        # Momentum handling
        if momentum != 0:
            if self.verbose_profile:
                t0 = time.perf_counter()
            # Partition indices into those with existing buffers and those
            # without.  We avoid per‑tensor branching inside the fast path.
            idx_with_buf = []
            idx_no_buf = []
            # Build separate lists for foreach operations
            for i, buf in enumerate(momentum_buffers):
                if buf is None:
                    idx_no_buf.append(i)
                else:
                    idx_with_buf.append(i)
            # Fast path for existing momentum buffers using foreach
            if idx_with_buf:
                bufs = [momentum_buffers[i] for i in idx_with_buf]
                grads_with_buf = [grads[i] for i in idx_with_buf]
                # buf = momentum * buf + (1 - dampening) * grad
                try:
                    torch._foreach_mul_(bufs, momentum)
                    torch._foreach_add_(bufs, grads_with_buf, alpha=1.0 - dampening)
                except Exception:
                    # fallback elementwise update
                    for j, i in enumerate(idx_with_buf):
                        bufs[j].mul_(momentum).add_(grads_with_buf[j], alpha=1.0 - dampening)
                # If not using Nesterov, replace grads with buf contents
                if not nesterov:
                    for idx, buf in zip(idx_with_buf, bufs):
                        grads[idx] = buf
                # Update original list of momentum_buffers
                for idx, buf in zip(idx_with_buf, bufs):
                    momentum_buffers[idx] = buf
            # Handle parameters without momentum buffers
            if idx_no_buf:
                for i in idx_no_buf:
                    # Clone gradient as the new momentum buffer
                    momentum_buffers[i] = grads[i].clone().detach()
                    if not nesterov:
                        grads[i] = momentum_buffers[i]
            # Apply Nesterov momentum: grad += momentum * buffer
            if nesterov:
                # Attempt a foreach update when all buffers have consistent
                # dtype/device; otherwise fall back to elementwise addition.
                try:
                    # Compute scaled momentum buffers: each element times the momentum factor
                    scaled_bufs = torch._foreach_mul([buf.clone() for buf in momentum_buffers], momentum)
                    updated_grads = torch._foreach_add(grads, scaled_bufs, alpha=1.0)
                    for i in range(len(grads)):
                        grads[i] = updated_grads[i]
                except Exception:
                    # Fallback: elementwise update
                    for i in range(len(grads)):
                        grads[i] = grads[i].add(momentum_buffers[i], alpha=momentum)
            if self.verbose_profile:
                t_momentum += time.perf_counter() - t0
        # Adaptive scaling per parameter
        if adaptive:
            # We attempt to use multi‑tensor operations to update the running
            # sum of squared gradients and scale the gradients.  This reduces
            # Python overhead compared to looping over each parameter.  The
            # basic idea is:
            #   1. Compute squared gradients for each tensor.
            #   2. For parameters that already have a running sum, add
            #      the squared gradient; otherwise, initialise the sum.
            #   3. Compute the per‑parameter scale as sqrt(sum) + eps.
            #   4. Divide each gradient by its corresponding scale.
            if self.verbose_profile:
                t0 = time.perf_counter()
            # Step 1: compute squared gradients for all tensors
            # We use elementwise multiplication of each grad by itself to
            # obtain grad**2.  Using foreach_mul with two lists computes
            # pairwise products in a single kernel.
            try:
                # torch._foreach_mul expects two lists of equal length and
                # performs elementwise multiplication: out[i] = a[i] * b[i].
                sq_grads = torch._foreach_mul(grads, grads)
            except Exception:
                # Fallback: compute squared gradients individually if
                # foreach_mul fails (e.g. due to heterogeneous dtypes)
                sq_grads = [g * g for g in grads]
            # Step 2: build a list of running sums and determine which
            # entries need updating.  We delay the addition until all
            # squared gradients are computed so we can batch the add.
            sum_sq_list = []  # List of running sums in order of params
            exist_sum_sq = []  # Sublist of existing sums to be updated
            exist_sq_grads = []  # Corresponding squared grads for updates
            for idx, (param, sq_grad) in enumerate(zip(params, sq_grads)):
                state = self.state[param]
                if 'sum_sq_grad' in state:
                    # Collect this sum and squared gradient for a batched
                    # addition below.  We will update the tensor in place.
                    exist_sum_sq.append(state['sum_sq_grad'])
                    exist_sq_grads.append(sq_grad)
                else:
                    # Initialize the running sum for this parameter.  We clone
                    # the squared gradient to avoid holding a reference to
                    # ``sq_grad`` which may be reused.
                    state['sum_sq_grad'] = sq_grad.clone()
                # Append the (possibly newly created) running sum to the full
                # list of sums to preserve the order corresponding to grads.
                sum_sq_list.append(state['sum_sq_grad'])
            # Update existing running sums in a single batched add if there
            # are any.  Using foreach_add_ performs in‑place addition on
            # each tensor: sum_sq += sq_grad.
            if exist_sum_sq:
                try:
                    torch._foreach_add_(exist_sum_sq, exist_sq_grads)
                except Exception:
                    # Fallback: elementwise addition if foreach_add fails
                    for s, g2 in zip(exist_sum_sq, exist_sq_grads):
                        s.add_(g2)
            # Step 3: compute scales = sqrt(sum_sq) + adaptive_eps
            try:
                scales = torch._foreach_sqrt(sum_sq_list)
            except Exception:
                scales = [s.sqrt() for s in sum_sq_list]
            # Add epsilon to each scale to avoid divide‑by‑zero
            try:
                torch._foreach_add_(scales, adaptive_eps)
            except Exception:
                for s in scales:
                    s.add_(adaptive_eps)
            # Step 4: divide gradients by scales elementwise.  Using
            # foreach_div_ avoids looping in Python.  If foreach_div_ fails,
            # we fall back to individual division.
            try:
                torch._foreach_div_(grads, scales)
            except Exception:
                for g, s in zip(grads, scales):
                    g.div_(s)
            if self.verbose_profile:
                t_adaptive += time.perf_counter() - t0
        # Apply learning rate and layer‑specific multipliers
        if self.verbose_profile:
            t0 = time.perf_counter()
        # Determine if per‑layer learning rate multipliers are used
        if group.get('layer_lr_multipliers'):
            # Apply parameter‑wise update with individual learning rates
            for param, grad in zip(params, grads):
                current_lr = lr
                layer_idx = self.param_to_layer.get(param)
                if layer_idx is not None:
                    current_lr *= group['layer_lr_multipliers'].get(layer_idx, 1.0)
                param.data.add_(grad, alpha=-current_lr)
        else:
            # Fast path: all parameters use the same learning rate, so use a
            # foreach update on the parameter data.  We update params in place
            # by adding the (negative) scaled gradients.
            try:
                # Use foreach for the parameter update
                torch._foreach_add_([p.data for p in params], grads, alpha=-lr)
            except Exception:
                # Fallback: individual parameter update
                for param, grad in zip(params, grads):
                    param.data.add_(grad, alpha=-lr)
        if self.verbose_profile:
            t_final_update += time.perf_counter() - t0
            # Print detailed profiling information for the update phase
            print(f"[UpdateProfiling] weight_decay={t_weight_decay:.6f}s, momentum={t_momentum:.6f}s, "
                  f"adaptive={t_adaptive:.6f}s, param_update={t_final_update:.6f}s")