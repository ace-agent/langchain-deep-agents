"""
CUDA kernel optimization agent powered by LangChain Deep Agents.

Stateless autoresearch loop: each iteration spawns a fresh agent with no
conversation history. State is persisted on disk (experiment_history.md,
results.tsv, submission.py) and the agent reads it via tools each cycle.

Usage:
    uv run agent.py                           # default: unlimited iterations, checkpoint every 5
    uv run agent.py --iterations 100          # stop after 100 iterations
    uv run agent.py --checkpoint-every 10     # print summary every 10 iterations
"""

import argparse
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from dotenv import load_dotenv

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langchain_openai import ChatOpenAI

import tools
from tools import (
    log_experiment,
    get_experiment_history,
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
    if not os.path.exists(tools.TSV_FILE):
        return "No experiments run yet."

    with open(tools.TSV_FILE) as f:
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
    """Build a fresh agent instance (no checkpointer — stateless per iteration)."""
    load_dotenv()

    model_name = os.environ.get("AUTORESEARCH_MODEL", "gpt-5.2")
    system_prompt = load_system_prompt()

    model = ChatOpenAI(
        model=model_name,
        use_responses_api=False,
        timeout=180,
        max_retries=3,
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
        backend=LocalShellBackend(root_dir=PROJECT_DIR, virtual_mode=True, env=env),
    )
    return agent


def log_conversation(run_dir: str, iteration: int, prompt: str, response: str) -> None:
    """Log the full conversation for each iteration for debugging."""
    conversation_log = os.path.join(run_dir, "conversation_log.md")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    
    with open(conversation_log, "a") as f:
        f.write(f"\n## Iteration {iteration} — {timestamp}\n\n")
        f.write("### Prompt:\n")
        f.write(f"```\n{prompt}\n```\n\n")
        f.write("### Response:\n")
        f.write(f"```\n{response}\n```\n\n")
        f.write("---\n")


def build_iteration_prompt(iteration: int, total_iterations: int, run_dir: str) -> str:
    """Build the single prompt the agent sees each iteration.

    The agent has no memory — it must reconstruct context from disk.
    We give it a concise summary in the prompt and tell it to use
    get_experiment_history + grep on experiment_history.md for details.

    The agent works on a COPY of submission.py inside the run directory,
    never touching the original.
    """
    results_summary = read_results_summary()
    is_first = iteration == 1

    # Agent sees paths relative to PROJECT_DIR (its root_dir)
    run_name = os.path.basename(run_dir)
    sub = f"runs/{run_name}/submission.py"
    results = f"runs/{run_name}/results.json"
    best = f"runs/{run_name}/best_submission.py"
    history = f"runs/{run_name}/experiment_history.md"

    if is_first:
        return (
            f"Iteration {iteration}/{total_iterations}.\n\n"
            "THIS IS THE FIRST ITERATION. YOU MUST ESTABLISH A BASELINE.\n\n"
            f"DO NOT MODIFY {sub}. DO NOT change ANY code.\n\n"
            "Execute these steps EXACTLY:\n"
            f"1. Read {sub} (just to capture its content for logging)\n"
            f"2. Run: python run_eval.py {sub} -o {results}\n"
            f"3. Read {results} to get the geomean timing\n"
            "4. Call log_experiment with:\n"
            f"   - kernel_code = the full content of {sub}\n"
            "   - hypothesis = 'Baseline test of expert kernel'\n"
            "   - time_us = the geomean time from results.json\n"
            "   - status = 'keep'\n\n"
            f"CRITICAL: Do NOT edit, modify, or change {sub} in any way.\n"
            "The purpose of this iteration is ONLY to measure current performance."
        )

    prior_context = ""
    if "No experiments run yet" not in results_summary:
        prior_context = f"\nCurrent experiment status:\n{results_summary}\n"

    return (
        f"Iteration {iteration}/{total_iterations}.\n"
        f"{prior_context}\n"
        f"WORKING FILE: {sub}\n"
        f"RESULTS FILE: {results}\n\n"
        "Instructions for this iteration:\n"
        "1. Call get_experiment_history to review the last few experiments "
        f"(hypotheses, results, what crashed). For older history, use grep on "
        f"{history}.\n"
        f"2. Read the current {sub}\n"
        "3. Based on what worked/failed before, form a NEW hypothesis\n"
        f"4. Implement ONE change to {sub}\n"
        f"5. Run `python run_eval.py {sub} -o {results}`\n"
        f"6. Read {results} and call log_experiment with the result\n\n"
        "RULES:\n"
        "- Make exactly ONE experiment per iteration\n"
        f"- If the result is worse than the best, revert {sub} to the best version "
        f"(copy from {best})\n"
        "- Do NOT repeat approaches that already crashed or produced worse results\n"
        f"- ONLY edit {sub} — do NOT modify any other files\n"
        "- Do NOT ask for instructions or summarize — just act"
    )


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
        print(f"  Plot updated: {tools.PLOT_FILE}")
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
        print(f"  Final plot saved to: {tools.PLOT_FILE}")
    except Exception:
        pass

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="GPU MODE Autoresearch Agent (Stateless Loop)")
    parser.add_argument("--iterations", "-n", type=int, default=50,
                        help="Total iterations to run (0 = unlimited, default: 50)")
    parser.add_argument("--checkpoint-every", "-c", type=int, default=5,
                        help="Print checkpoint summary every N iterations (default: 5)")
    args = parser.parse_args()

    load_dotenv()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment or .env file")
        sys.exit(1)

    # Create timestamped run directory
    gpu_target = os.environ.get("GPU_TARGET", "B200")
    leaderboard = os.environ.get("LEADERBOARD", "nvfp4_group_gemm")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{gpu_target}_{leaderboard}"
    run_dir = os.path.join(PROJECT_DIR, "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    set_run_directory(run_dir)

    # Copy submission.py into the run directory — agent works on this copy only
    import shutil
    original_submission = os.path.join(PROJECT_DIR, "submission.py")
    run_submission = os.path.join(run_dir, "submission.py")
    shutil.copy2(original_submission, run_submission)

    total_iterations = args.iterations
    checkpoint_every = args.checkpoint_every

    model_name = os.environ.get("AUTORESEARCH_MODEL", "gpt-5.2")
    starting_iteration = _get_next_iteration() - 1

    print(f"Starting kernel optimization agent (stateless loop)...")
    print(f"  Model: {model_name}")
    print(f"  GPU Target: {gpu_target}")
    print(f"  Leaderboard: {leaderboard}")
    print(f"  Run directory: {run_dir}")
    print(f"  Iterations: {'unlimited' if total_iterations == 0 else total_iterations}")
    print(f"  Checkpoint every: {checkpoint_every} iterations")
    print(f"  Prior experiments: {starting_iteration}")
    print()

    start_time = time.time()
    iteration = 0
    consecutive_failures = 0
    max_consecutive_failures = 10

    try:
        while True:
            iteration += 1

            if total_iterations > 0 and iteration > total_iterations:
                break

            print(f"\n{'='*60}")
            print(f"  Agent iteration {iteration}{'/' + str(total_iterations) if total_iterations > 0 else ''}")
            print(f"{'='*60}\n")

            # Fresh agent each iteration — no accumulated context
            agent = build_agent()
            config = {"configurable": {"thread_id": f"iter-{iteration}"}}

            prompt = build_iteration_prompt(
                iteration,
                total_iterations if total_iterations > 0 else iteration,
                run_dir,
            )

            try:
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": prompt}]},
                    config=config,
                )

                n_msgs = len(result["messages"])
                final = result["messages"][-1]
                content = (
                    final.content
                    if hasattr(final, "content")
                    else str(final)
                )
                
                # Log the full conversation for debugging
                log_conversation(run_dir, iteration, prompt, content or "(empty)")
                
                print(f"\n--- Agent completed (used {n_msgs} messages) ---")
                print(content[:500] if content else "(empty)")

                # The agent's final text can be empty if its last action was a
                # tool call (e.g. log_experiment). That's fine — it still did work.
                # Only count as failure if the agent barely did anything (<=3 msgs
                # means it didn't even make a single tool call).
                if n_msgs > 3:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    print(f"  WARNING: Agent did no work ({consecutive_failures}/{max_consecutive_failures})")

            except Exception as e:
                consecutive_failures += 1
                error_msg = f"Agent framework error: {e}\n{traceback.format_exc()}"
                
                # Log the error conversation
                log_conversation(run_dir, iteration, prompt, f"ERROR: {error_msg}")
                
                print(f"\n--- Agent framework error: {e} ({consecutive_failures}/{max_consecutive_failures}) ---")
                traceback.print_exc()

            if consecutive_failures >= max_consecutive_failures:
                print(f"\n--- Too many consecutive failures, stopping ---")
                break

            # Checkpoint
            if iteration % checkpoint_every == 0:
                print_checkpoint(
                    iteration,
                    total_iterations if total_iterations > 0 else iteration,
                    start_time,
                )

    except KeyboardInterrupt:
        print(f"\n\n--- Interrupted by user at iteration {iteration} ---")
    finally:
        print_final_report(
            total_iterations if total_iterations > 0 else iteration,
            iteration,
            start_time,
        )


if __name__ == "__main__":
    main()
