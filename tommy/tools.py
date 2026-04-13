import subprocess
from langchain_core.tools import tool


@tool
def read_file(path: str) -> str:
    """Read a file from disk and return its contents."""
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return f"ERROR: file not found: {path}"
    except Exception as e:
        return f"ERROR: {e}"


@tool
def edit_file(path: str, old_str: str, new_str: str) -> str:
    """Replace old_str with new_str in a file. old_str must match exactly once."""
    try:
        content = open(path).read()
        count = content.count(old_str)
        if count == 0:
            return f"ERROR: old_str not found in {path}"
        if count > 1:
            return f"ERROR: old_str found {count} times in {path} — must match exactly once"
        open(path, "w").write(content.replace(old_str, new_str))
        return "ok"
    except Exception as e:
        return f"ERROR: {e}"


@tool
def run_bash(cmd: str) -> str:
    """Run a shell command and return stdout + stderr. Timeout 600s."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=600
        )
        output = result.stdout + result.stderr
        return output if output.strip() else "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out after 600s"
    except Exception as e:
        return f"ERROR: {e}"