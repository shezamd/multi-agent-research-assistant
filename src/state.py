"""
Shared state definition for the Multi-Agent Research Assistant.

In LangGraph, a 'state' is a TypedDict that flows through every node
in the graph. Each node receives the full state, does its work, and
returns a dict containing only the keys it wants to update.

LangGraph merges those partial updates back into the state automatically
before passing it to the next node.
"""

import operator
from typing import Annotated

from typing_extensions import TypedDict


class ResearchState(TypedDict):
    """The shared whiteboard that all agents read from and write to.

    Parameters
    ----------
    question : str
        The original research question from the user. Never mutated
        after the graph starts.
    tasks : list[str]
        Sub-tasks produced by the Planner (e.g. ["Find X", "Explain Y"]).
        Populated once; read by the parallel Researcher nodes.
    research_results : list[str]
        One summary string per Researcher. We use Annotated + operator.add
        so that parallel nodes can each *append* their result rather than
        overwriting each other.  LangGraph merges concurrent writes with
        the reducer function (operator.add for lists = concatenation).
    draft_answer : str
        The Writer's synthesised answer before Critic review.
    feedback : str
        The Critic's notes. An empty string signals approval.
    final_answer : str
        Set only once the Critic is satisfied. This is what we print.
    """

    question: str
    tasks: list[str]
    # operator.add is the reducer: parallel nodes append, not overwrite
    research_results: Annotated[list[str], operator.add]
    draft_answer: str
    feedback: str
    final_answer: str
    revision_count: int  # incremented each time the Critic loops back to Writer
