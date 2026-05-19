#!POPCORN leaderboard nvfp4_group_gemm
#!POPCORN gpu B200

import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    abc_tensors, _sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes = data

    outs = []
    for (a, b, c), (sfa_r, sfb_r), (_m, _n, _k, l) in zip(
        abc_tensors, sfasfb_reordered_tensors, problem_sizes
    ):
        for li in range(l):
            # reordered scales are [32,4,rest_mn,4,rest_k,L]
            scale_a = sfa_r[..., li].permute(1, 0, 2, 3, 4).contiguous().view(-1)
            scale_b = sfb_r[..., li].permute(1, 0, 2, 3, 4).contiguous().view(-1)
            torch._scaled_mm(
                a[..., li],
                b[..., li].transpose(0, 1),
                scale_a,
                scale_b,
                out=c[..., li],
                out_dtype=torch.float16,
            )
        outs.append(c)
    return outs
