import re
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage

from tools import read_file, edit_file, run_bash


# what we expect back from the judge after each run
class ExperimentDecision(BaseModel):
    keep: bool
    val_bpb: float
    peak_vram_mb: float
    reason: str
    next_idea: str


# two separate models — agent does the work, judge makes the call
agent_llm = ChatOpenAI(model="gpt-4o", max_tokens=8192)
judge_llm = ChatOpenAI(model="gpt-4o", max_tokens=1024).with_structured_output(ExperimentDecision)

TOOLS = [read_file, edit_file, run_bash]
TOOL_MAP = {t.name: t for t in TOOLS}
agent_llm = agent_llm.bind_tools(TOOLS)


def execute_tool_calls(response) -> list:
    tool_messages = []
    for tc in response.tool_calls:
        result = TOOL_MAP[tc["name"]].invoke(tc["args"])
        tool_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return tool_messages


def make_decision(run_log: str, description: str) -> ExperimentDecision:
    prompt = f"""
You just ran a 5-minute ML training experiment.
Here is the output log:

{run_log}

The experiment was: {description}

Parse val_bpb and peak_vram_mb from the log.
Set keep=true only if val_bpb improved (lower than previous best).
If the run crashed or didn't improve, keep=false.
Give a one-sentence reason and suggest what to try next.
"""
    try:
        return judge_llm.invoke(prompt)
    except Exception as e:
        # something went wrong parsing the log, treat as a failed run
        return ExperimentDecision(
            keep=False,
            val_bpb=0.0,
            peak_vram_mb=0.0,
            reason=f"Could not parse results: {e}",
            next_idea="Check the log for errors and try a simpler change.",
        )


def log_result(commit: str, decision: ExperimentDecision, description: str):
    memory_gb = round(decision.peak_vram_mb / 1024, 1)
    status = "keep" if decision.keep else "discard"
    val_bpb_str = f"{decision.val_bpb:.6f}" if decision.val_bpb else "0.000000"
    row = f"{commit}\t{val_bpb_str}\t{memory_gb}\t{status}\t{description}\n"
    with open("results.tsv", "a") as f:
        f.write(row)
    print(f"[logged] {status} | bpb={val_bpb_str} | {description}")


def make_messages(program: str, context: str) -> list:
    # fresh message list every experiment — no tool call history to worry about
    return [
        SystemMessage(content=program),
        HumanMessage(content=(
            "Setup is already done. Branch is created, results.tsv is initialized. "
            "Here is the current context from previous experiments:\n\n"
            f"{context}\n\n"
            "Now run the next experiment: read train.py and results.tsv, make one focused "
            "code change, commit it, then run: run_bash('uv run train.py > run.log 2>&1'). "
            "When done, tell me: "
            "1) the git commit hash (run 'git rev-parse --short HEAD') "
            "2) a short description of what you changed "
            "3) the full contents of run.log. "
            "Then stop and wait."
        )),
    ]


def run_agent():
    program = open("program.md").read()
    last_decision = None
    iteration = 0

    while True:
        iteration += 1
        print(f"\n{'='*60}\nExperiment #{iteration}\n{'='*60}")

        # build context summary from last decision to carry forward
        if last_decision:
            context = (
                f"Last experiment: {last_decision.reason} "
                f"(keep={last_decision.keep}, bpb={last_decision.val_bpb:.6f}). "
                f"Suggested next idea: {last_decision.next_idea}"
            )
        else:
            context = "This is the first experiment. Start by reading results.tsv and train.py."

        # fresh message list each experiment
        messages = make_messages(program, context)

        # let the agent keep calling tools until it's done
        while True:
            response = agent_llm.invoke(messages)
            messages.append(response)
            if response.tool_calls:
                messages.extend(execute_tool_calls(response))
            else:
                break

        summary = response.content if isinstance(response.content, str) else str(response.content)

        commit_match = re.search(r'\b([0-9a-f]{7})\b', summary)
        commit = commit_match.group(1) if commit_match else "unknown"

        # prefer reading run.log directly over parsing the agent's summary
        try:
            run_log = open("run.log").read()
        except FileNotFoundError:
            run_log = summary

        desc_match = re.search(r'description[:\s]+([^\n\.]+)', summary, re.IGNORECASE)
        description = desc_match.group(1).strip() if desc_match else summary[:60].strip()

        print(f"[deciding] commit={commit}")
        decision = make_decision(run_log, description)
        print(f"[decision] keep={decision.keep} | bpb={decision.val_bpb:.6f} | {decision.reason}")

        if decision.keep:
            print("[keeping]")
        else:
            print("[discarding]")
            run_bash.invoke({"cmd": "git reset --hard HEAD~1"})

        log_result(commit, decision, description)
        last_decision = decision


if __name__ == "__main__":
    run_agent()