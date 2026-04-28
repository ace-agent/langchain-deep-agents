"""
Modal app for evaluating CUDA kernels on a B200 GPU.

The remote function receives kernel source code as a string, dynamically loads
it, runs correctness tests against the reference (a @ b), then benchmarks with
warmup + timed iterations using CUDA events for precise measurement.

Returns a JSON string to avoid pickle/torch deserialization issues on the client.
"""

import modal

app = modal.App("cuda-kernel-eval")

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "triton")
)


@app.function(gpu="B200", image=gpu_image, timeout=600)
def evaluate_kernel(
    kernel_code: str,
    warmup_iters: int = 5,
    eval_iters: int = 10,
) -> str:
    """
    Run correctness tests and benchmarks for a CUDA kernel on a B200.

    Returns a JSON string to avoid requiring torch on the caller side.
    """
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

    CORRECTNESS_CASES = [
        {"m": 64, "n": 64, "k": 64, "seed": 53124},
        {"m": 128, "n": 128, "k": 128, "seed": 3321},
        {"m": 256, "n": 256, "k": 256, "seed": 1200},
        {"m": 32, "n": 512, "k": 32, "seed": 32523},
        {"m": 64, "n": 1024, "k": 64, "seed": 4327},
    ]
    BENCHMARK_CASE = {"m": 4096, "n": 5120, "k": 4096, "seed": 123456}

    def generate_input(m, n, k, seed):
        gen = torch.Generator(device="cuda")
        gen.manual_seed(seed)
        a = torch.empty(m, k, device="cuda", dtype=torch.float16)
        a.uniform_(0, 1, generator=gen)
        b = torch.empty(k, n, device="cuda", dtype=torch.float16)
        b.uniform_(0, 1, generator=gen)
        c = torch.empty(m, n, device="cuda", dtype=torch.float16)
        return (a, b, c)

    def load_kernel(code):
        task_mod = types.ModuleType("task")
        task_mod.input_t = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        task_mod.output_t = torch.Tensor
        sys.modules["task"] = task_mod

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
        m, n, k, seed = case["m"], case["n"], case["k"], case["seed"]
        detail = {"m": m, "n": n, "k": k, "seed": seed, "passed": False, "error": None}
        try:
            data = generate_input(m, n, k, seed)
            ref = data[0] @ data[1]

            data2 = generate_input(m, n, k, seed)
            out = custom_kernel(data2)
            torch.cuda.synchronize()

            if torch.allclose(out, ref, atol=1e-1, rtol=1e-1):
                detail["passed"] = True
                result["tests_passed"] += 1
            else:
                max_diff = (out - ref).abs().max().item()
                detail["error"] = f"Mismatch: max_diff={max_diff:.6f}"
        except Exception as e:
            detail["error"] = str(e)

        result["test_details"].append(detail)

    if result["tests_passed"] < result["tests_total"]:
        result["error"] = "Testing failed"
        return json.dumps(result)

    if eval_iters <= 0:
        result["success"] = True
        return json.dumps(result)

    try:
        bm = BENCHMARK_CASE
        m, n, k, seed = bm["m"], bm["n"], bm["k"], bm["seed"]

        for _ in range(warmup_iters):
            data = generate_input(m, n, k, seed)
            _ = custom_kernel(data)
            torch.cuda.synchronize()

        timings_us = []
        for _ in range(eval_iters):
            data = generate_input(m, n, k, seed)

            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)

            start_evt.record()
            _ = custom_kernel(data)
            end_evt.record()
            torch.cuda.synchronize()

            elapsed_ms = start_evt.elapsed_time(end_evt)
            timings_us.append(elapsed_ms * 1000.0)

        mean_us = sum(timings_us) / len(timings_us)
        variance = sum((t - mean_us) ** 2 for t in timings_us) / len(timings_us)
        std_us = math.sqrt(variance)
        stderr_us = std_us / math.sqrt(len(timings_us))
        min_us = min(timings_us)
        max_us = max(timings_us)

        result["benchmark"] = {
            "m": m,
            "n": n,
            "k": k,
            "seed": seed,
            "mean_us": round(mean_us, 1),
            "std_us": round(std_us, 2),
            "stderr_us": round(stderr_us, 1),
            "min_us": round(min_us, 1),
            "max_us": round(max_us, 1),
        }
        result["success"] = True

    except Exception:
        result["error"] = f"Benchmark failed:\n{traceback.format_exc()}"

    return json.dumps(result)
