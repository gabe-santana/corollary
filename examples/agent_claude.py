"""The same task with a real model: Claude, through the official Anthropic SDK.

    pip install "corollary[anthropic]"
    export ANTHROPIC_API_KEY=...        # or `ant auth login`
    python examples/agent_claude.py

The script makes a handful of API calls (one per agent step, plus one per re-derived belief).
"""

from __future__ import annotations

from corollary import Agent, AnthropicModel, tool

FILINGS = {"Q1": 4.0e9, "Q2": 4.3e9, "Q3": 4.5e9, "Q4": 4.8e9}


@tool(trust="high")
def get_revenue(quarter: str) -> float:
    """Quarterly revenue in USD from SEC filings. quarter is one of Q1, Q2, Q3, Q4."""
    return FILINGS[quarter]


def main() -> None:
    agent = Agent(AnthropicModel("claude-opus-5-5"), tools=[get_revenue])
    report = agent.run("Compare Q2 and Q3 revenue and assess the growth trend.")

    print("ANSWER:", report.answer)
    for rejection in report.rejections:
        print("  rejected:", rejection)
    if not report.completed:
        print("\nThe agent did not answer:", report.error or f"no answer within {agent.max_steps} steps")
        return
    print("\nPROOF:\n" + report.proof.render())
    print("\n" + str(report.verify()))

    print("\n--- Q2 revenue is restated to $4.1B ---\n")
    agent.kb.correct("revenue:Q2", 4.1e9, reason="restated in 10-K/A")
    for change in agent.repair():
        print(change)
    print("\nREPAIRED ANSWER:", report.answer)
    print(report.verify())


if __name__ == "__main__":
    main()
