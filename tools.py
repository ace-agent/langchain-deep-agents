"""Custom tools for the autoresearch deep agent."""

import os
import re
from langchain.tools import tool


@tool
def parse_training_output(log_path: str) -> dict:
    """Parse the training output log file and extract metrics.

    Reads the log file produced by `uv run train.py > run.log 2>&1` and
    extracts the summary block that appears after the `---` separator.

    Args:
        log_path: Path to the training log file (e.g. "run.log").

    Returns:
        A dict with keys like val_bpb, training_seconds, total_seconds,
        peak_vram_mb, mfu_percent, total_tokens_M, num_steps, num_params_M,
        depth. If the run crashed, returns {"crashed": True, "error": "..."}.
    """
    if not os.path.exists(log_path):
        return {"crashed": True, "error": f"Log file not found: {log_path}"}

    with open(log_path, "r") as f:
        content = f.read()

    if "FAIL" in content:
        return {"crashed": True, "error": "Training failed (loss exploded or NaN)"}

    # Find the --- separator and parse key: value lines after it
    parts = content.split("---")
    if len(parts) < 2:
        tail = content[-2000:] if len(content) > 2000 else content
        return {"crashed": True, "error": f"No summary block found. Tail:\n{tail}"}

    summary_text = parts[-1]
    metrics = {}
    for line in summary_text.strip().splitlines():
        line = line.strip()
        match = re.match(r"^(\w+):\s+(.+)$", line)
        if match:
            key, value = match.group(1), match.group(2)
            try:
                metrics[key] = float(value)
            except ValueError:
                metrics[key] = value

    if "val_bpb" not in metrics:
        return {"crashed": True, "error": f"val_bpb not found in summary. Got: {metrics}"}

    metrics["crashed"] = False
    return metrics


@tool
def log_experiment(
    commit: str,
    val_bpb: float,
    memory_gb: float,
    status: str,
    description: str,
) -> str:
    """Log an experiment result to results.tsv.

    Appends a row to results.tsv (tab-separated). Creates the file with
    headers if it doesn't exist. Call this after every experiment run.

    Args:
        commit: Short git commit hash (7 chars).
        val_bpb: The val_bpb metric achieved. Use 0.0 for crashes.
        memory_gb: Peak VRAM in GB (peak_vram_mb / 1024). Use 0.0 for crashes.
        status: One of "keep", "discard", or "crash".
        description: Short text describing what this experiment tried.

    Returns:
        Confirmation message with the logged row.
    """
    tsv_path = "results.tsv"
    header = "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"

    if not os.path.exists(tsv_path):
        with open(tsv_path, "w") as f:
            f.write(header)

    row = f"{commit}\t{val_bpb:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n"
    with open(tsv_path, "a") as f:
        f.write(row)

    return f"Logged: {row.strip()}"
