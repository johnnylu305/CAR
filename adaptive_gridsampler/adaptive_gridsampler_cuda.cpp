#include <ATen/ATen.h>
#include <torch/extension.h>

#include "adaptive_gridsampler_kernel.cuh"

int adaptive_gridsampler_cuda_forward(at::Tensor& img, at::Tensor& kernels, at::Tensor& offsets_h, at::Tensor& offsets_v, int offset_unit, int padding, at::Tensor& output)
{
    adaptive_gridsampler_kernel_forward(img, kernels, offsets_h, offsets_v, offset_unit, padding, output);
    return 1;
}

int adaptive_gridsampler_cuda_backward(at::Tensor& img, at::Tensor& kernels, at::Tensor& offsets_h, at::Tensor& offsets_v, int offset_unit, int padding, at::Tensor& output, at::Tensor& K, at::Tensor& H, at::Tensor& V)
{
    adaptive_gridsampler_kernel_backward(img, kernels, offsets_h, offsets_v, offset_unit, padding,
 output, K, H, V);
    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &adaptive_gridsampler_cuda_forward, "adaptive gridsampler forward (CUDA)");
    m.def("backward", &adaptive_gridsampler_cuda_backward, "adaptive gridsampler backward (CUDA)");
}
