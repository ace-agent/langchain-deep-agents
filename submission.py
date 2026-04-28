import torch
from task import input_t, output_t


def generate_input(m: int, n: int, k: int, seed: int) -> input_t:
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)
    a = torch.empty(m, k, device='cuda', dtype=torch.float16)
    a.uniform_(0, 1, generator=gen)
    b = torch.empty(k, n, device='cuda', dtype=torch.float16)
    b.uniform_(0, 1, generator=gen)
    c = torch.empty(m, n, device='cuda', dtype=torch.float16)
    return a, b, c


def ref_kernel(data: input_t) -> output_t:
    """Reference implementation using PyTorch's optimized matmul."""
    a, b, c = data
    return a @ b


def custom_kernel(data: input_t) -> output_t:
    """
    Custom kernel implementation - starts as copy of reference.
    The agent will optimize this function to beat PyTorch's cuBLAS performance.
    """
    a, b, c = data
    return a @ b