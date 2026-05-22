"""Custom tools for the CUDA kernel optimization deep agent."""

import os
from datetime import datetime, timezone
from langchain.tools import tool

# Default files (will be overridden by set_run_directory)
HISTORY_FILE = "experiment_history.md"
TSV_FILE = "results.tsv"
PLOT_FILE = "progress.png"

# Run directory - set by agent.py
_run_directory = None

def set_run_directory(run_dir):
    """Set the directory where experiment files should be stored for this run."""
    global _run_directory, HISTORY_FILE, TSV_FILE, PLOT_FILE
    _run_directory = run_dir
    HISTORY_FILE = os.path.join(run_dir, "experiment_history.md")
    TSV_FILE = os.path.join(run_dir, "results.tsv")
    PLOT_FILE = os.path.join(run_dir, "progress.png")


def _ensure_history_file():
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w") as f:
            f.write("# Experiment History\n\n")
            f.write("This file tracks every kernel attempt, its code, and results.\n")
            f.write("The agent reads this before each new attempt to avoid repeating failures.\n\n")


def _ensure_tsv_file():
    if not os.path.exists(TSV_FILE):
        with open(TSV_FILE, "w") as f:
            f.write("iteration\tcommit\ttime_us\tstatus\tdescription\n")


def _get_next_iteration() -> int:
    """Read the TSV and return the next iteration number."""
    if not os.path.exists(TSV_FILE):
        return 1
    with open(TSV_FILE) as f:
        lines = f.readlines()
    data_lines = [l for l in lines[1:] if l.strip()]
    return len(data_lines) + 1


def _update_plot():
    """Regenerate progress.png from results.tsv after every logged experiment."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    if not os.path.exists(TSV_FILE):
        return

    with open(TSV_FILE) as f:
        lines = f.readlines()

    if len(lines) < 2:
        return

    header = lines[0].strip().split("\t")
    has_iteration_col = header[0] == "iteration"

    iterations = []
    times = []
    statuses = []
    best_times = []
    best_so_far = float("inf")

    for i, line in enumerate(lines[1:], start=1):
        parts = line.strip().split("\t")
        if not parts or len(parts) < 3:
            continue

        try:
            if has_iteration_col and len(parts) >= 5:
                it = int(parts[0]) if parts[0].isdigit() else i
                time_us = float(parts[2])
                status = parts[3]
            elif has_iteration_col and len(parts) >= 4:
                it = int(parts[0]) if parts[0].isdigit() else i
                time_us = float(parts[1])
                status = parts[2]
            else:
                it = i
                time_us = float(parts[1]) if len(parts) > 1 else 0.0
                status = parts[2] if len(parts) > 2 else "unknown"
        except (ValueError, IndexError):
            continue

        iterations.append(it)
        times.append(time_us if time_us > 0 else None)
        statuses.append(status)

        if time_us > 0 and time_us < best_so_far:
            best_so_far = time_us
        best_times.append(best_so_far if best_so_far < float("inf") else None)

    if not iterations:
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    keep_x = [it for it, s, t in zip(iterations, statuses, times) if s == "keep" and t]
    keep_y = [-t for s, t in zip(statuses, times) if s == "keep" and t]  # Negative latency
    discard_x = [it for it, s, t in zip(iterations, statuses, times) if s == "discard" and t]
    discard_y = [-t for s, t in zip(statuses, times) if s == "discard" and t]  # Negative latency
    crash_x = [it for it, s in zip(iterations, statuses) if s == "crash"]

    all_valid = [-t for t in times if t and t > 0]  # Negative latency
    if all_valid:
        y_lo = min(all_valid) * 1.15  # More negative (worse)
        y_hi = max(all_valid) * 0.85  # Less negative (better)
    else:
        y_lo, y_hi = -100, 0

    # Plot baseline (iteration 1) first, regardless of status
    baseline_iteration = 1
    baseline_status = None
    baseline_time = None
    
    if baseline_iteration in iterations:
        idx = iterations.index(baseline_iteration)
        baseline_status = statuses[idx] 
        baseline_time = times[idx]
        
        if baseline_status == "keep" and baseline_time and baseline_time > 0:
            ax.scatter([baseline_iteration], [-baseline_time], c="#22c55e", s=120, zorder=7, label="baseline", edgecolors="gold", linewidths=2, marker="^")
        elif baseline_status == "discard" and baseline_time and baseline_time > 0:
            ax.scatter([baseline_iteration], [-baseline_time], c="#ef4444", s=120, zorder=7, label="baseline", edgecolors="gold", linewidths=2, marker="^")
        elif baseline_status == "crash":
            # For crashed baseline, show it at the bottom of the plot
            crash_y = y_lo if 'y_lo' in locals() else -100
            ax.scatter([baseline_iteration], [crash_y], c="#fbbf24", s=120, zorder=7, label="baseline (crash)", edgecolors="gold", linewidths=2, marker="^")

    if keep_y:
        # Plot regular keep points (excluding baseline which is handled above)
        regular_keep_x = [x for x in keep_x if x != baseline_iteration]
        regular_keep_y = [y for x, y in zip(keep_x, keep_y) if x != baseline_iteration]
        
        if regular_keep_x:
            ax.scatter(regular_keep_x, regular_keep_y, c="#22c55e", s=60, zorder=5, label="keep", edgecolors="white", linewidths=0.5)
    
    if discard_y:
        # Plot regular discard points (excluding baseline which is handled above)
        regular_discard_x = [x for x in discard_x if x != baseline_iteration]
        regular_discard_y = [y for x, y in zip(discard_x, discard_y) if x != baseline_iteration]
        
        if regular_discard_x:
            ax.scatter(regular_discard_x, regular_discard_y, c="#ef4444", s=40, zorder=4, label="discard", edgecolors="white", linewidths=0.5, alpha=0.7)

    if crash_x:
        # Plot regular crashes (excluding baseline which is handled above)
        regular_crash_x = [x for x in crash_x if x != baseline_iteration]
        
        if regular_crash_x:
            ax.scatter(regular_crash_x, [y_lo] * len(regular_crash_x), c="#fbbf24", s=25, zorder=3, label=f"crash ({len(regular_crash_x)})", marker="x", alpha=0.6)

    valid_best = [(it, -bt) for it, bt in zip(iterations, best_times) if bt is not None]  # Negative best times
    if valid_best:
        bx, by = zip(*valid_best)
        ax.step(bx, by, where="post", color="#3b82f6", linewidth=2, label="best time", zorder=6)

    ax.set_ylim(y_lo, y_hi)
    
    # Set x-axis to show full expected iteration range (0 to ~50)
    if iterations:
        max_iter = max(iterations)
        # Show full range: start at 0, end at reasonable total (50) or current max + buffer
        expected_total = max(50, max_iter + 10)  # At least 50, or current progress + buffer
        ax.set_xlim(0, expected_total)
        # Set reasonable tick spacing based on range
        if expected_total <= 20:
            tick_spacing = 2
        elif expected_total <= 50:
            tick_spacing = 5
        else:
            tick_spacing = 10
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Negative Latency (-μs)", fontsize=12)
    leaderboard = os.environ.get("LEADERBOARD", "nvfp4_group_gemm")
    ax.set_title(f"GPU MODE {leaderboard} — Autoresearch Progress", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    if all_valid and best_so_far < float("inf"):
        ax.annotate(
            f"Best: {best_so_far:.1f} μs",
            xy=(0.02, 0.98), xycoords="axes fraction",  # Top-left since higher is better now
            fontsize=11, fontweight="bold", color="#3b82f6",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#3b82f6", alpha=0.9),
        )

    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=150)
    plt.close(fig)


@tool
def log_experiment(
    kernel_code: str,
    hypothesis: str,
    time_us: float,
    status: str,
    error_message: str = "",
    commit: str = "HEAD",
) -> str:
    """Log a kernel experiment to both experiment_history.md and results.tsv.

    Call this after every submission attempt. The markdown log captures the full
    kernel code so the agent can review what worked and what didn't.
    Automatically assigns an iteration number and updates the progress plot.

    Args:
        kernel_code: The full contents of submission.py that was tested.
        hypothesis: What this experiment was trying to achieve and why.
        time_us: Execution time in microseconds. Use 0.0 for crashes.
        status: One of "keep" (new best), "discard" (worse than best), or "crash".
        error_message: If status is "crash", the error output. Empty string otherwise.
        commit: Short git commit hash (7 chars). Defaults to "HEAD".

    Returns:
        Confirmation message with iteration number.
    """
    _ensure_history_file()
    _ensure_tsv_file()

    iteration = _get_next_iteration()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    status_emoji = {"keep": "✅", "discard": "❌", "crash": "💥"}.get(status, "❓")

    with open(HISTORY_FILE, "a") as f:
        f.write(f"---\n\n")
        f.write(f"## Experiment #{iteration} — {timestamp} {status_emoji} {status.upper()}\n\n")
        f.write(f"**Hypothesis:** {hypothesis}\n\n")
        if status == "crash":
            f.write(f"**Result:** CRASH\n\n")
            if error_message:
                f.write(f"**Error:**\n```\n{error_message[:2000]}\n```\n\n")
        else:
            f.write(f"**Result:** {time_us:.2f} μs\n\n")
        f.write(f"**Kernel code:**\n```python\n{kernel_code}\n```\n\n")

    desc = hypothesis[:100]
    with open(TSV_FILE, "a") as f:
        f.write(f"{iteration}\t{commit}\t{time_us:.2f}\t{status}\t{desc}\n")

    try:
        _update_plot()
        plot_msg = f" (plot updated: {os.path.basename(PLOT_FILE)})"
    except Exception as e:
        plot_msg = f" (plot update failed: {e})"

    if status == "keep" and _run_directory and kernel_code.strip():
        best_path = os.path.join(_run_directory, "best_submission.py")
        with open(best_path, "w") as f:
            f.write(kernel_code)

    if status == "crash":
        return f"Logged iteration #{iteration} CRASH: {hypothesis}{plot_msg}"
    return f"Logged iteration #{iteration} {status}: {time_us:.2f} μs — {hypothesis}{plot_msg}"


@tool
def get_experiment_history(last_n: int = 10) -> str:
    """Read recent experiment history from experiment_history.md.

    Returns the last N experiments (hypothesis, result, kernel code).
    For older experiments, use grep on experiment_history.md or read results.tsv.

    Args:
        last_n: Number of most recent experiments to return. Defaults to 10.

    Returns:
        The recent experiment entries, or a message if no history exists yet.
    """
    if not os.path.exists(HISTORY_FILE):
        return "No experiment history yet. This will be the first run."

    with open(HISTORY_FILE, "r") as f:
        content = f.read()

    # Split into individual experiment sections
    sections = content.split("---\n\n## Experiment #")
    if len(sections) <= 1:
        # No experiments logged yet, or only the header
        if len(content) < 500:
            return content
        return content[-30000:]

    header = sections[0]
    experiments = sections[1:]

    # Return last N experiments
    recent = experiments[-last_n:]
    result = header.rstrip() + "\n\n"
    if len(experiments) > last_n:
        result += f"[... {len(experiments) - last_n} earlier experiments omitted — use grep to search ...]\n\n"
    for exp in recent:
        result += f"---\n\n## Experiment #{exp}"

    # Safety cap
    if len(result) > 80000:
        result = result[-80000:]

    return result
