#!/usr/bin/env python3
"""
CLI wrapper that submits a kernel to the Modal B200 evaluator and writes
results.json in the same markdown format the agent already parses.

Usage:
    python run_eval.py submission.py -o results.json
    python run_eval.py submission.py -o results.json --mode test   # correctness only
"""

import argparse
import json
import sys

import modal


def format_results_markdown(res: dict, mode: str = "leaderboard") -> str:
    gpu = res.get("gpu_name", "NVIDIA B200")
    torch_ver = res.get("torch_version", "unknown")
    plat = res.get("platform", "unknown")

    if res["success"]:
        status_line = f"**B200 on Modal ✅ success**"
    elif res.get("error") and "Testing failed" in res["error"]:
        status_line = f"**B200 on Modal ❌ failure**"
    else:
        status_line = f"**B200 on Modal ❌ failure**"

    lines = [status_line]

    if res["success"]:
        lines.append("> ✅ Testing successful")
        lines.append("> ✅ Benchmarking successful")
    elif res["tests_passed"] == res["tests_total"]:
        lines.append("> ✅ Testing successful")
        lines.append("> ❌ Benchmarking failed")
    else:
        lines.append("> ❌ Testing failed")

    lines.append("")
    lines.append("Running on:")
    lines.append(f"* GPU: `{gpu}`")
    lines.append(f"* Device count: `1`")
    lines.append(f"* Runtime: `CUDA`")
    lines.append(f"* Platform: `{plat}`")
    lines.append(f"* Torch: `{torch_ver}`")
    lines.append(f"* Hostname: `modal`")
    lines.append("")

    passed = res["tests_passed"]
    total = res["tests_total"]
    lines.append(f"## {'✅' if passed == total else '❌'} Passed {passed}/{total} tests:")
    lines.append("```")
    for td in res["test_details"]:
        icon = "✅" if td["passed"] else "❌"
        lines.append(f"{icon} g: {td['g']}; m: {td['m']}; n: {td['n']}; k: {td['k']}")
    lines.append("```")

    if res.get("error") and not res["success"]:
        lines.append("")
        lines.append(f"## Error:")
        lines.append("```")
        lines.append(res["error"])
        lines.append("```")

    bm = res.get("benchmark")
    if bm and mode == "leaderboard":
        lines.append("")
        lines.append("## Benchmarks:")
        lines.append("```")
        lines.append(f"Geometric mean: ⏱ {bm['geomean_us']} µs")
        lines.append("")
        for bd in res.get("benchmark_details", []):
            lines.append(
                f"  Case {bd['case_idx']}: g={bd['g']} "
                f"⏱ {bd['mean_us']} ± {bd['stderr_us']} µs "
                f"⚡ {bd['min_us']} µs 🐌 {bd['max_us']} µs"
            )
        lines.append("```")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a CUDA kernel on Modal B200")
    parser.add_argument("submission", help="Path to submission.py")
    parser.add_argument("-o", "--output", default="results.json", help="Output results file")
    parser.add_argument(
        "--mode",
        choices=["test", "leaderboard"],
        default="leaderboard",
        help="'test' for correctness only, 'leaderboard' for correctness + benchmark",
    )
    args = parser.parse_args()

    try:
        with open(args.submission) as f:
            kernel_code = f.read()
    except FileNotFoundError:
        print(f"Error: {args.submission} not found")
        sys.exit(1)

    print(f"Submitting {args.submission} to Modal B200 ({args.mode} mode)...")

    evaluate_kernel = modal.Function.from_name("cuda-kernel-eval", "evaluate_kernel")

    if args.mode == "test":
        raw = evaluate_kernel.remote(kernel_code, warmup_iters=0, eval_iters=0)
    else:
        raw = evaluate_kernel.remote(kernel_code)

    res = json.loads(raw)
    md = format_results_markdown(res, mode=args.mode)
    with open(args.output, "w") as f:
        json.dump(md, f)

    print(md)

    if res["success"]:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
