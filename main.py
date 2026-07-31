"""
Multi-Agent Research Assistant — CLI entry point.

Usage
-----
    uv run python main.py "Your research question here"
    uv run python main.py  # prompts for input if no argument given

LangSmith
---------
Tracing is enabled automatically via the LANGSMITH_TRACING=true env var
loaded from .env.  After each run you can view:
  - The full graph execution timeline
  - Each agent's prompt, response, and token usage
  - Latency per node
at https://smith.langchain.com under the project name set in .env.

We also pass a run_name and run_id to .invoke() so each run is easy to
find in the LangSmith UI by name rather than by timestamp.
"""

import sys
import uuid
import os

from dotenv import load_dotenv

# Load .env before importing any LangChain/LangGraph modules.
# LangSmith reads LANGSMITH_TRACING and LANGSMITH_API_KEY at import time,
# so this must be the very first thing we do.
load_dotenv()

from langchain_core.runnables import RunnableConfig  # noqa: E402
from src.graph import app  # noqa: E402


def main() -> None:
    """Run the research assistant for a single question.

    Reads the question from the first CLI argument, or prompts the user
    if none is provided.  Invokes the compiled LangGraph app, prints a
    progress log as each agent runs, and finally prints the answer.
    """
    # Get question from CLI arg or interactive prompt
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Enter your research question: ").strip()

    if not question:
        print("Error: question cannot be empty.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")

    # A unique run_id lets us construct a direct LangSmith trace link.
    # LangGraph forwards this ID to every child span automatically.
    run_id = str(uuid.uuid4())
    project = os.getenv("LANGSMITH_PROJECT", "default")

    config = RunnableConfig(
        run_id=run_id,
        run_name="research-assistant",  # displayed as the trace title in LangSmith
        tags=["research-assistant"],    # filterable in LangSmith's UI
    )

    # Initial state — provide safe defaults for every field.
    # TypedDict doesn't enforce defaults at runtime, so we set them here
    # to avoid KeyError inside nodes that call state.get(...).
    initial_state = {
        "question": question,
        "tasks": [],
        "research_results": [],
        "draft_answer": "",
        "feedback": "",
        "final_answer": "",
        "revision_count": 0,
    }

    # Invoke the graph — this blocks until END is reached.
    # The print() calls inside each agent node give live progress feedback.
    final_state = app.invoke(initial_state, config=config)

    # Print the answer
    print(f"\n{'='*60}")
    print("FINAL ANSWER")
    print(f"{'='*60}")
    print(final_state["final_answer"] or final_state["draft_answer"])

    # LangSmith trace link (only meaningful if tracing is enabled)
    if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
        print(f"\n{'='*60}")
        print(f"LangSmith trace:")
        print(f"  Project : {project}")
        print(f"  Run ID  : {run_id}")
        print(f"  URL     : https://smith.langchain.com (filter by run ID above)")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
