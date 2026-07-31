"""
Mock search tool used by Researcher agents.

A real implementation would call a search API (Tavily, SerpAPI, etc.).
The mock returns plausible-sounding fake snippets so we can develop and
test the graph without any external API calls or costs.

Keeping tools separate from agents makes it easy to swap the mock for a
real search function later without touching the Researcher code.
"""

import hashlib


def mock_search(query: str, num_results: int = 3) -> list[str]:
    """Simulate a web search and return a list of result snippets.

    Parameters
    ----------
    query : str
        The search query string.
    num_results : int, optional
        Number of fake snippets to return, by default 3.

    Returns
    -------
    list[str]
        A list of fake but plausible search result snippets.

    Notes
    -----
    Results are deterministic: the same query always returns the same
    snippets. This makes the mock useful for reproducible tests.
    """
    # Use a hash of the query to seed deterministic "results"
    seed = int(hashlib.md5(query.encode()).hexdigest(), 16) % 10_000

    templates = [
        f"According to recent studies on '{query}', researchers have found "
        f"significant evidence suggesting that this topic has broad implications "
        f"across multiple disciplines. (Source: Journal of Research, vol. {seed % 50 + 1})",

        f"A comprehensive review of '{query}' indicates that the current "
        f"consensus among experts points to {seed % 5 + 2} key factors that "
        f"drive outcomes in this area. (Source: ResearchGate, {2020 + seed % 5})",

        f"Historical data related to '{query}' shows a consistent trend over "
        f"the past {seed % 20 + 5} years, with notable acceleration since "
        f"{2015 + seed % 8}. (Source: DataCommons, accessed 2025)",

        f"Experts disagree on aspects of '{query}': one school of thought "
        f"emphasises practical applications while another focuses on theoretical "
        f"foundations. Both perspectives have merit. (Source: Nature, {seed % 10 + 2014})",

        f"Case studies examining '{query}' in real-world contexts reveal that "
        f"implementation challenges are often underestimated. Success requires "
        f"careful planning and iterative refinement. (Source: HBR, {seed % 6 + 2018})",
    ]

    # Pick num_results snippets pseudo-randomly based on the seed
    indices = [(seed + i * 7) % len(templates) for i in range(num_results)]
    return [templates[i] for i in indices]
