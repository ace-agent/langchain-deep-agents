"""
Autoresearch agent powered by LangChain Deep Agents.

Usage:
    uv run agent.py                          # uses default model (openai:gpt-4o)
    AUTORESEARCH_MODEL=openai:o3 uv run agent.py  # override model

The agent autonomously modifies train.py, runs experiments, and tracks
results in results.tsv. It uses git to version control each experiment.
"""

import os
import sys
from dotenv import load_dotenv

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langgraph.checkpoint.memory import MemorySaver

from tools import parse_training_output, log_experiment

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_system_prompt() -> str:
    program_path = os.path.join(PROJECT_DIR, "program.md")
    with open(program_path, "r") as f:
        return f.read()


def create_agent():
    # Load environment variables from .env file
    load_dotenv()
    
    model = os.environ.get("AUTORESEARCH_MODEL", "openai:gpt-4o")
    system_prompt = load_system_prompt()
    checkpointer = MemorySaver()

    # Set up environment for LocalShellBackend to use the virtual environment
    venv_path = os.path.join(PROJECT_DIR, ".venv", "bin")
    current_path = os.environ.get("PATH", "")
    env = {
        "PATH": f"{venv_path}:{current_path}",
        "VIRTUAL_ENV": os.path.join(PROJECT_DIR, ".venv"),
        "PYTHONPATH": PROJECT_DIR,
    }
    # Copy other important env vars
    for key in ["OPENAI_API_KEY", "AUTORESEARCH_MODEL"]:
        if key in os.environ:
            env[key] = os.environ[key]

    agent = create_deep_agent(
        model=model,
        tools=[parse_training_output, log_experiment],
        system_prompt=system_prompt,
        backend=LocalShellBackend(root_dir=PROJECT_DIR, virtual_mode=False, env=env),
        checkpointer=checkpointer,
    )
    return agent


def main():
    # Load .env file at startup
    load_dotenv()
    
    # Check if API key is available
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment or .env file")
        print("Please add OPENAI_API_KEY=\"your-key-here\" to your .env file")
        sys.exit(1)
    
    agent = create_agent()

    kickoff_message = (
        "Read program.md, then read train.py and prepare.py for full context. "
        "Set up a new experiment branch, establish the baseline by running "
        "train.py as-is, then begin the autonomous experiment loop."
    )

    print(f"Starting autoresearch agent...")
    print(f"  Model: {os.environ.get('AUTORESEARCH_MODEL', 'openai:gpt-4o')}")
    print(f"  Project dir: {PROJECT_DIR}")
    print(f"  Kick-off: {kickoff_message}")
    print()

    config = {"configurable": {"thread_id": "autoresearch-main"}}

    result = agent.invoke(
        {"messages": [{"role": "user", "content": kickoff_message}]},
        config=config,
    )

    final_message = result["messages"][-1]
    print("\n--- Agent finished ---")
    print(final_message.content if hasattr(final_message, "content") else str(final_message))


if __name__ == "__main__":
    main()
