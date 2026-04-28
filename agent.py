"""
CUDA kernel optimization agent powered by LangChain Deep Agents.

Runs autoresearch for X iterations, checkpointing every Y iterations to
measure whether the optimization loop is actually improving performance.

Usage:
    uv run agent.py                           # default: 50 iterations, checkpoint every 5
    uv run agent.py --iterations 100          # 100 iterations
    uv run agent.py --checkpoint-every 10     # benchmark every 10 iterations
    uv run agent.py --iterations 100 --checkpoint-every 10

The agent autonomously modifies submission.py, submits to the leaderboard,
and tracks results in experiment_history.md. All prior results and exploration
traces are fed back to the agent at each iteration.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from dotenv import load_dotenv

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from tools import (
    log_experiment,
    get_experiment_history,
    TSV_FILE,
    PLOT_FILE,
    _update_plot,
    _get_next_iteration,
    set_run_directory,
)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_system_prompt() -> str:
    with open(os.path.join(PROJECT_DIR, "program.md")) as f:
        return f.read()


def read_results_summary() -> str:
    """Build a concise summary of all experiments so far for the agent context."""
    if not os.path.exists(TSV_FILE):
        return "No experiments run yet."

    with open(TSV_FILE) as f:
        lines = f.readlines()

    if len(lines) < 2:
        return "No experiments run yet."

    total = len(lines) - 1
    keeps = []
    discards = 0
    crashes = 0
    best_time = float("inf")
    best_desc = ""
    last_5 = []

    for line in lines[1:]:
        parts = line.strip().split("\t")
        if len(parts) < 5:
            continue
        it, commit, time_str, status, desc = parts[0], parts[1], parts[2], parts[3], parts[4]
        try:
            t = float(time_str)
        except ValueError:
            t = 0.0

        if status == "keep" and t > 0:
            keeps.append((int(it) if it.isdigit() else 0, t, desc))
            if t < best_time:
                best_time = t
                best_desc = desc
        elif status == "discard":
            discards += 1
        elif status == "crash":
            crashes += 1

        last_5.append(f"  #{it}: {t:.1f}μs ({status}) — {desc[:60]}")

    last_5 = last_5[-5:]

    summary = f"=== EXPERIMENT SUMMARY ({total} total) ===\n"
    summary += f"Best time: {best_time:.1f} μs" if best_time < float("inf") else "Best time: none yet"
    if best_desc:
        summary += f" — {best_desc[:80]}\n"
    else:
        summary += "\n"
    summary += f"Keeps: {len(keeps)} | Discards: {discards} | Crashes: {crashes}\n"
    if keeps:
        summary += "Keep history (iteration -> time):\n"
        for it, t, d in keeps[-10:]:
            summary += f"  #{it}: {t:.1f}μs — {d[:60]}\n"
    summary += f"\nLast 5 experiments:\n" + "\n".join(last_5) + "\n"
    return summary


def build_agent():
    load_dotenv()

    model_name = os.environ.get("AUTORESEARCH_MODEL", "gpt-4o")
    system_prompt = load_system_prompt()
    checkpointer = MemorySaver()

    model = ChatOpenAI(
        model=model_name,
        use_responses_api=False,
        timeout=120,
        max_retries=2,
    )

    venv_path = os.path.join(PROJECT_DIR, ".venv", "bin")
    current_path = os.environ.get("PATH", "")
    env = {
        "PATH": f"{venv_path}:{current_path}",
        "VIRTUAL_ENV": os.path.join(PROJECT_DIR, ".venv"),
        "PYTHONPATH": PROJECT_DIR,
    }
    for key in ["OPENAI_API_KEY", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "GPU_TARGET", "LEADERBOARD"]:
        if key in os.environ:
            env[key] = os.environ[key]

    agent = create_deep_agent(
        model=model,
        tools=[log_experiment, get_experiment_history],
        system_prompt=system_prompt,
        backend=LocalShellBackend(root_dir=PROJECT_DIR, virtual_mode=False, env=env),
        checkpointer=checkpointer,
    )
    return agent


def print_checkpoint(iteration: int, total_iterations: int, start_time: float):
    """Print a checkpoint summary with stats and timing."""
    elapsed = time.time() - start_time
    elapsed_min = elapsed / 60
    rate = iteration / elapsed_min if elapsed_min > 0 else 0

    summary = read_results_summary()

    print(f"\n{'#'*60}")
    print(f"  CHECKPOINT — Iteration {iteration}/{total_iterations}")
    print(f"  Elapsed: {elapsed_min:.1f} min | Rate: {rate:.1f} iter/min")
    print(f"{'#'*60}")
    print(summary)

    try:
        _update_plot()
        print(f"  Plot updated: {PLOT_FILE}")
    except Exception as e:
        print(f"  Plot update failed: {e}")

    print(f"{'#'*60}\n")


def print_final_report(total_iterations: int, actual_iterations: int, start_time: float):
    """Print the final report when the run completes or is interrupted."""
    elapsed = time.time() - start_time
    elapsed_min = elapsed / 60

    summary = read_results_summary()

    print(f"\n{'='*60}")
    print(f"  FINAL REPORT")
    print(f"{'='*60}")
    print(f"  Iterations completed: {actual_iterations}/{total_iterations}")
    print(f"  Total time: {elapsed_min:.1f} min")
    print(f"{'='*60}")
    print(summary)

    try:
        _update_plot()
        print(f"  Final plot saved to: {PLOT_FILE}")
    except Exception:
        pass

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="GPU MODE Autoresearch Agent")
    parser.add_argument("--iterations", "-n", type=int, default=50,
                        help="Total number of agent iterations to run (default: 50)")
    parser.add_argument("--checkpoint-every", "-c", type=int, default=5,
                        help="Print checkpoint summary every N iterations (default: 5)")
    args = parser.parse_args()

    load_dotenv()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment or .env file")
        sys.exit(1)

    # Create timestamped run directory
    gpu_target = os.environ.get("GPU_TARGET", "B200")
    leaderboard = os.environ.get("LEADERBOARD", "matmul_v2")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{gpu_target}_{leaderboard}"
    run_dir = os.path.join(PROJECT_DIR, "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Set the run directory for tools to use
    set_run_directory(run_dir)

    total_iterations = args.iterations
    checkpoint_every = args.checkpoint_every

    agent = build_agent()

    model_name = os.environ.get("AUTORESEARCH_MODEL", "gpt-4o")
    starting_iteration = _get_next_iteration() - 1

    print(f"Starting kernel optimization agent...")
    print(f"  Model: {model_name}")
    print(f"  GPU Target: {gpu_target}")
    print(f"  Leaderboard: {leaderboard}")
    print(f"  Run directory: {run_dir}")
    print(f"  Total iterations: {total_iterations}")
    print(f"  Checkpoint every: {checkpoint_every} iterations")
    print(f"  Prior experiments: {starting_iteration}")
    print()

    config = {"configurable": {"thread_id": "kernel-opt-main"}}
    start_time = time.time()

    kickoff_message = (
        "Read program.md for full instructions. Then call get_experiment_history "
        "to review any prior attempts. Read the current submission.py. "
        "Then begin the autonomous optimization loop: propose a hypothesis, "
        "implement it, submit, log the result, and repeat.\n\n"
        + read_results_summary()
    )

    iteration = 0
    try:
        msg = kickoff_message
        while iteration < total_iterations:
            iteration += 1

            print(f"\n{'='*60}")
            print(f"  Agent iteration {iteration}/{total_iterations}")
            print(f"{'='*60}\n")

            result = agent.invoke(
                {"messages": [{"role": "user", "content": msg}]},
                config=config,
            )

            n_msgs = len(result["messages"])
            final = result["messages"][-1]
            content = (
                final.content
                if hasattr(final, "content")
                else str(final)
            )
            print(f"\n--- Agent yielded ({n_msgs} messages) ---")
            print(content[:500] if content else "(empty)")

            # Checkpoint every Y iterations
            if iteration % checkpoint_every == 0:
                print_checkpoint(iteration, total_iterations, start_time)

            # Build rich context for next iteration with recent results
            recent_summary = read_results_summary()
            msg = (
                f"Continue the optimization loop. Iteration {iteration + 1}/{total_iterations}.\n\n"
                f"{recent_summary}\n"
                "Call get_experiment_history for full prior code traces. "
                "Then propose and implement the next experiment. "
                "Do not summarize or ask for instructions — just act."
            )

    except KeyboardInterrupt:
        print(f"\n\n--- Interrupted by user at iteration {iteration} ---")
    except Exception as e:
        print(f"\n--- Agent error at iteration {iteration}: {e} ---")
        import traceback
        traceback.print_exc()
    finally:
        print_final_report(total_iterations, iteration, start_time)


if __name__ == "__main__":
    main()
