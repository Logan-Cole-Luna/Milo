#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// CUDA kernel for fused gradient normalization
__global__ void fused_gradient_normalization_kernel(
    float* __restrict__ gradients,
    const float* __restrict__ scale_factors,
    const int n_params,
    const int* __restrict__ param_sizes,
    const int* __restrict__ param_offsets,
    const int* __restrict__ layer_indices,
    const float eps,
    const float scale_factor,
    const bool scale_aware,
    const bool layer_wise
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= n_params) return;
    
    int param_idx = tid;
    int param_size = param_sizes[param_idx];
    int param_offset = param_offsets[param_idx];
    int layer_idx = layer_indices[param_idx];
    
    // Shared memory for reduction
    __shared__ float sdata[256];
    int local_tid = threadIdx.x;
    
    // Compute norm for this parameter or layer
    float local_sum = 0.0f;
    
    if (layer_wise) {
        // Layer-wise normalization: compute norm across all parameters in the layer
        for (int i = local_tid; i < param_size; i += blockDim.x) {
            float val = gradients[param_offset + i];
            local_sum += val * val;
        }
    } else {
        // Parameter-wise normalization
        for (int i = local_tid; i < param_size; i += blockDim.x) {
            float val = gradients[param_offset + i];
            local_sum += val * val;
        }
    }
    
    // Reduction in shared memory
    sdata[local_tid] = local_sum;
    __syncthreads();
    
    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (local_tid < stride) {
            sdata[local_tid] += sdata[local_tid + stride];
        }
        __syncthreads();
    }
    
    // Compute norm and apply normalization
    if (local_tid == 0) {
        float norm = sqrtf(sdata[0] + eps);
        float normalization_factor = 1.0f / norm;
        
        // Apply normalization to gradients
        for (int i = 0; i < param_size; i++) {
            int grad_idx = param_offset + i;
            float grad_val = gradients[grad_idx];
            
            if (scale_aware) {
                float clamped_norm = fminf(norm, 1.0f);
                gradients[grad_idx] = scale_factor * grad_val + 
                                     (1.0f - scale_factor) * clamped_norm * grad_val * normalization_factor;
            } else {
                gradients[grad_idx] = grad_val * normalization_factor;
            }
        }
    }
}

// Enhanced momentum kernel with fused operations
__global__ void enhanced_momentum_kernel(
    float* __restrict__ momentum_long,
    float* __restrict__ momentum_short,
    const float* __restrict__ gradients,
    const float momentum_long_decay,
    const float momentum_short_decay,
    const int n_elements
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n_elements) {
        float grad = gradients[tid];
        momentum_long[tid] = momentum_long_decay * momentum_long[tid] + (1.0f - momentum_long_decay) * grad;
        momentum_short[tid] = momentum_short_decay * momentum_short[tid] + (1.0f - momentum_short_decay) * grad;
    }
}

// Wrapper functions for Python interface
torch::Tensor fused_gradient_normalization(
    torch::Tensor gradients,
    torch::Tensor scale_factors,
    torch::Tensor param_sizes,
    torch::Tensor param_offsets,
    torch::Tensor layer_indices,
    float eps,
    float scale_factor,
    bool scale_aware,
    bool layer_wise
) {
    auto grad_flat = gradients.contiguous();
    int n_params = param_sizes.size(0);
    
    const int threads = 256;
    const int blocks = (n_params + threads - 1) / threads;
    
    fused_gradient_normalization_kernel<<<blocks, threads>>>(
        grad_flat.data_ptr<float>(),
        scale_factors.data_ptr<float>(),
        n_params,
        param_sizes.data_ptr<int>(),
        param_offsets.data_ptr<int>(),
        layer_indices.data_ptr<int>(),
        eps,
        scale_factor,
        scale_aware,
        layer_wise
    );
    
    return grad_flat;
}

torch::Tensor enhanced_momentum_update(
    torch::Tensor momentum_long,
    torch::Tensor momentum_short,
    torch::Tensor gradients,
    float momentum_long_decay,
    float momentum_short_decay
) {
    int n_elements = gradients.numel();
    
    const int threads = 256;
    const int blocks = (n_elements + threads - 1) / threads;
    
    enhanced_momentum_kernel<<<blocks, threads>>>(
        momentum_long.data_ptr<float>(),
        momentum_short.data_ptr<float>(),
        gradients.data_ptr<float>(),
        momentum_long_decay,
        momentum_short_decay,
        n_elements
    );
    
    return momentum_long;
}
