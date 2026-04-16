import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

cuda_src = r"""
#include <cuda_runtime.h>
__global__ void sum_reduce(const float* input, float* output, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float mySum = (i < n) ? input[i] : 0.0f;
    sdata[tid] = mySum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}
"""

cpp_src = "torch::Tensor launch_sum(torch::Tensor input);"

_module = load_inline(
    name='vectorsum_cuda',
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=['sum_reduce'],
    verbose=False,
)

def custom_kernel(data: input_t) -> output_t:
    input_tensor, output_tensor = data
    n = input_tensor.numel()
    threads = 256
    blocks = (n + threads - 1) // threads
    partial = torch.zeros(blocks, device=input_tensor.device, dtype=input_tensor.dtype)
    _module.sum_reduce(input_tensor, partial, n)
    output_tensor[0] = partial.sum()
    return output_tensor
