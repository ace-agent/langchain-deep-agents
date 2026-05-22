"""
Modal app for evaluating NVFP4 Group GEMM kernels on a B200 GPU.

The remote function receives kernel source code as a string, dynamically loads
it, runs correctness tests against the reference (torch._scaled_mm), then
benchmarks with warmup + timed iterations using CUDA events.

Returns a JSON string to avoid pickle/torch deserialization issues on the client.
"""

import modal

app = modal.App("cuda-kernel-eval")

gpu_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04",
        add_python="3.11",
    )
    .entrypoint([])
    .apt_install("ninja-build")
    .pip_install("torch", "numpy", "triton", "ninja")
    .env({
        "TORCH_CUDA_ARCH_LIST": "10.0a",
        "CUDA_HOME": "/usr/local/cuda",
    })
)


@app.function(gpu="B200", image=gpu_image, timeout=600, min_containers=1)
def evaluate_kernel(
    kernel_code: str,
    warmup_iters: int = 20,
    eval_iters: int = 50,
) -> str:
    import torch
    import platform
    import math
    import json
    import traceback
    import importlib
    import importlib.util
    import sys
    import types
    import tempfile
    import os

    sf_vec_size = 16

    def ceil_div(a, b):
        return (a + b - 1) // b

    def _create_fp4_tensors(l, mn, k):
        ref_i8 = torch.randint(255, size=(l, mn, k // 2), dtype=torch.uint8, device="cuda")
        ref_i8 = ref_i8 & 0b1011_1011
        return ref_i8.permute(1, 2, 0).view(torch.float4_e2m1fn_x2)

    def create_reordered_scale_factor_tensor(l, mn, k, ref_f8_tensor):
        sf_k = ceil_div(k, sf_vec_size)
        atom_m = (32, 4)
        atom_k = 4
        mma_shape = (
            l,
            ceil_div(mn, atom_m[0] * atom_m[1]),
            ceil_div(sf_k, atom_k),
            atom_m[0],
            atom_m[1],
            atom_k,
        )
        mma_permute_order = (3, 4, 1, 5, 2, 0)
        rand_int_tensor = torch.randint(1, 3, mma_shape, dtype=torch.int8, device='cuda')
        reordered_f8_tensor = rand_int_tensor.to(dtype=torch.float8_e4m3fn)
        reordered_f8_tensor = reordered_f8_tensor.permute(*mma_permute_order)

        if ref_f8_tensor.device.type == 'cpu':
            ref_f8_tensor = ref_f8_tensor.cuda()

        i_idx = torch.arange(mn, device='cuda')
        j_idx = torch.arange(sf_k, device='cuda')
        b_idx = torch.arange(l, device='cuda')

        i_grid, j_grid, b_grid = torch.meshgrid(i_idx, j_idx, b_idx, indexing='ij')

        mm = i_grid // (atom_m[0] * atom_m[1])
        mm32 = i_grid % atom_m[0]
        mm4 = (i_grid % 128) // atom_m[0]
        kk = j_grid // atom_k
        kk4 = j_grid % atom_k

        reordered_f8_tensor[mm32, mm4, mm, kk4, kk, b_grid] = ref_f8_tensor[i_grid, j_grid, b_grid]

        return reordered_f8_tensor

    def to_blocked(input_matrix):
        rows, cols = input_matrix.shape
        n_row_blocks = ceil_div(rows, 128)
        n_col_blocks = ceil_div(cols, 4)
        padded_rows = n_row_blocks * 128
        padded_cols = n_col_blocks * 4
        if padded_rows != rows or padded_cols != cols:
            padded = torch.nn.functional.pad(
                input_matrix,
                (0, padded_cols - cols, 0, padded_rows - rows),
                mode="constant", value=0,
            )
        else:
            padded = input_matrix
        blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
        rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
        return rearranged.flatten()

    def generate_input(m_list, n_list, k_list, g, seed):
        torch.manual_seed(seed)
        abc_tensors = []
        sfasfb_tensors = []
        sfasfb_reordered_tensors = []
        problem_sizes = []
        l = 1
        for group_idx in range(g):
            mi, ni, ki = m_list[group_idx], n_list[group_idx], k_list[group_idx]
            a_ref = _create_fp4_tensors(l, mi, ki)
            b_ref = _create_fp4_tensors(l, ni, ki)
            c_ref = torch.randn((l, mi, ni), dtype=torch.float16, device="cuda").permute(1, 2, 0)

            sf_k = ceil_div(ki, sf_vec_size)
            sfa_ref_cpu = torch.randint(1, 3, (l, mi, sf_k), dtype=torch.int8).to(
                dtype=torch.float8_e4m3fn
            ).permute(1, 2, 0)
            sfb_ref_cpu = torch.randint(1, 3, (l, ni, sf_k), dtype=torch.int8).to(
                dtype=torch.float8_e4m3fn
            ).permute(1, 2, 0)

            sfa_reordered = create_reordered_scale_factor_tensor(l, mi, ki, sfa_ref_cpu)
            sfb_reordered = create_reordered_scale_factor_tensor(l, ni, ki, sfb_ref_cpu)

            abc_tensors.append((a_ref, b_ref, c_ref))
            sfasfb_tensors.append((sfa_ref_cpu, sfb_ref_cpu))
            sfasfb_reordered_tensors.append((sfa_reordered, sfb_reordered))
            problem_sizes.append((mi, ni, ki, l))
        return (abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes)

    def ref_kernel(data):
        abc_tensors, sfasfb_tensors, _, problem_sizes = data
        result_tensors = []
        for i, ((a_ref, b_ref, c_ref), (sfa_ref, sfb_ref), (m, n, k, l)) in enumerate(
            zip(abc_tensors, sfasfb_tensors, problem_sizes)
        ):
            for l_idx in range(l):
                scale_a = to_blocked(sfa_ref[:, :, l_idx])
                scale_b = to_blocked(sfb_ref[:, :, l_idx])
                res = torch._scaled_mm(
                    a_ref[:, :, l_idx].view(torch.float4_e2m1fn_x2),
                    b_ref[:, :, l_idx].transpose(0, 1).view(torch.float4_e2m1fn_x2),
                    scale_a.cuda(),
                    scale_b.cuda(),
                    bias=None,
                    out_dtype=torch.float16,
                )
                c_ref[:, :, l_idx] = res
            result_tensors.append(c_ref)
        return result_tensors

    CORRECTNESS_CASES = [
        {"g": 2, "m": [128, 256], "n": [256, 256], "k": [256, 256], "seed": 42},
        {"g": 2, "m": [128, 384], "n": [4096, 4096], "k": [1536, 1536], "seed": 100},
        {"g": 2, "m": [192, 320], "n": [3072, 3072], "k": [4096, 4096], "seed": 200},
    ]

    BENCHMARK_CASES = [
        {
            "g": 8,
            "m": [80, 176, 128, 72, 64, 248, 96, 160],
            "n": [4096]*8,
            "k": [7168]*8,
            "seed": 1001,
        },
        {
            "g": 8,
            "m": [40, 76, 168, 72, 164, 148, 196, 160],
            "n": [7168]*8,
            "k": [2048]*8,
            "seed": 1002,
        },
        {
            "g": 2,
            "m": [192, 320],
            "n": [3072, 3072],
            "k": [4096, 4096],
            "seed": 1003,
        },
        {
            "g": 2,
            "m": [128, 384],
            "n": [4096, 4096],
            "k": [1536, 1536],
            "seed": 1004,
        },
    ]

    def load_kernel(code):
        task_mod = types.ModuleType("task")
        task_mod.input_t = tuple
        task_mod.output_t = list
        sys.modules["task"] = task_mod

        utils_mod = types.ModuleType("utils")
        def make_match_reference(ref_fn, rtol=1e-3, atol=1e-3):
            def checker(output, expected):
                for o, e in zip(output, expected):
                    if not torch.allclose(o, e, rtol=rtol, atol=atol):
                        return False
                return True
            return checker
        utils_mod.make_match_reference = make_match_reference
        sys.modules["utils"] = utils_mod

        tmp_dir = tempfile.mkdtemp()
        path = os.path.join(tmp_dir, "submission.py")
        with open(path, "w") as f:
            f.write(code)

        spec = importlib.util.spec_from_file_location("submission", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["submission"] = mod
        spec.loader.exec_module(mod)

        if not hasattr(mod, "custom_kernel"):
            raise AttributeError("submission.py must define a `custom_kernel` function")
        return mod.custom_kernel

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"

    result = {
        "success": False,
        "tests_passed": 0,
        "tests_total": len(CORRECTNESS_CASES),
        "test_details": [],
        "benchmark": None,
        "benchmark_details": [],
        "gpu_name": gpu_name,
        "torch_version": str(torch.__version__),
        "platform": platform.platform(),
        "error": None,
    }

    try:
        custom_kernel = load_kernel(kernel_code)
    except Exception:
        result["error"] = f"Failed to load kernel:\n{traceback.format_exc()}"
        return json.dumps(result)

    for case in CORRECTNESS_CASES:
        g = case["g"]
        detail = {"g": g, "m": case["m"], "n": case["n"], "k": case["k"], "passed": False, "error": None}
        try:
            ref_data = generate_input(case["m"], case["n"], case["k"], g, case["seed"])
            ref_out = ref_kernel(ref_data)

            test_data = generate_input(case["m"], case["n"], case["k"], g, case["seed"])
            custom_out = custom_kernel(test_data)
            torch.cuda.synchronize()

            all_match = True
            max_diff = 0.0
            for ref_c, cust_c in zip(ref_out, custom_out):
                if not torch.allclose(ref_c, cust_c, atol=1e-1, rtol=1e-1):
                    diff = (ref_c - cust_c).abs().max().item()
                    max_diff = max(max_diff, diff)
                    all_match = False

            if all_match:
                detail["passed"] = True
                result["tests_passed"] += 1
            else:
                detail["error"] = f"Mismatch: max_diff={max_diff:.6f}"
        except Exception as e:
            detail["error"] = f"{type(e).__name__}: {str(e)}"

        result["test_details"].append(detail)

    if result["tests_passed"] < result["tests_total"]:
        result["error"] = "Testing failed"
        return json.dumps(result)

    if eval_iters <= 0:
        result["success"] = True
        return json.dumps(result)

    try:
        import time
        case_times = []
        case_medians = []

        for bi, bcase in enumerate(BENCHMARK_CASES):
            g = bcase["g"]

            # Warmup phase
            for _ in range(warmup_iters):
                data = generate_input(bcase["m"], bcase["n"], bcase["k"], g, bcase["seed"])
                _ = custom_kernel(data)
                torch.cuda.synchronize()

            # Small cooldown to let clocks/thermals stabilize after warmup burst
            torch.cuda.synchronize()
            time.sleep(0.01)

            timings_us = []
            for _ in range(eval_iters):
                data = generate_input(bcase["m"], bcase["n"], bcase["k"], g, bcase["seed"])
                
                # Clear cache for consistent allocator state
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)

                start_evt.record()
                _ = custom_kernel(data)
                end_evt.record()
                torch.cuda.synchronize()

                elapsed_ms = start_evt.elapsed_time(end_evt)
                timings_us.append(elapsed_ms * 1000.0)

            # Sort for percentile calculations
            timings_us.sort()
            n = len(timings_us)
            
            mean_us = sum(timings_us) / n
            median_us = timings_us[n // 2]
            variance = sum((t - mean_us) ** 2 for t in timings_us) / n
            std_us = math.sqrt(variance)
            stderr_us = std_us / math.sqrt(n)
            min_us = timings_us[0]
            max_us = timings_us[-1]
            p10_us = timings_us[n // 10]
            p90_us = timings_us[int(n * 0.9)]

            case_detail = {
                "case_idx": bi,
                "g": g,
                "m": bcase["m"],
                "n": bcase["n"],
                "k": bcase["k"],
                "mean_us": round(mean_us, 1),
                "median_us": round(median_us, 1),
                "std_us": round(std_us, 2),
                "stderr_us": round(stderr_us, 1),
                "min_us": round(min_us, 1),
                "max_us": round(max_us, 1),
                "p10_us": round(p10_us, 1),
                "p90_us": round(p90_us, 1),
            }
            result["benchmark_details"].append(case_detail)
            case_times.append(mean_us)
            case_medians.append(median_us)

        geomean_mean = math.exp(sum(math.log(t) for t in case_times) / len(case_times))
        geomean_median = math.exp(sum(math.log(t) for t in case_medians) / len(case_medians))

        result["benchmark"] = {
            "geomean_us": round(geomean_mean, 1),
            "geomean_median_us": round(geomean_median, 1),
            "case_times_us": [round(t, 1) for t in case_times],
            "case_medians_us": [round(t, 1) for t in case_medians],
            "num_cases": len(case_times),
        }
        result["success"] = True

    except Exception:
        result["error"] = f"Benchmark failed:\n{traceback.format_exc()}"

    return json.dumps(result)
