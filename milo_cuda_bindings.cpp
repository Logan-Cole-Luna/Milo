#include <torch/extension.h>

// Forward declarations
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
);

torch::Tensor enhanced_momentum_update(
    torch::Tensor momentum_long,
    torch::Tensor momentum_short,
    torch::Tensor gradients,
    float momentum_long_decay,
    float momentum_short_decay
);

// Python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_gradient_normalization", &fused_gradient_normalization, "Fused gradient normalization");
    m.def("enhanced_momentum_update", &enhanced_momentum_update, "Enhanced momentum update");
}
