"""An agent run end to end with a scripted model, so it works with no API key.

    python examples/agent_offline.py

The scripted responses are exactly what a real model would send under the claim contract.
Watch for three things:

1. The runtime, not the model, executes tools and records their results.
2. A claim that cheats (uses a tool result it has not seen yet) is rejected and fed back.
3. After a correction, the answer goes OUT, and `agent.repair()` re-derives it from the
   current beliefs only. The retracted figure never reaches the model again.
"""

from __future__ import annotations

from corollary import Agent, ScriptedModel, tool

FILINGS = {"Q2": 4.3e9, "Q3": 4.5e9}
GROWTH = "({revenue:Q3} - {revenue:Q2}) / {revenue:Q2} * 100"


@tool(trust="high")
def get_revenue(quarter: str) -> float:
    """Quarterly revenue in USD, from SEC filings."""
    return FILINGS[quarter]


def call(quarter: str) -> dict:  # type: ignore[type-arg]
    return {"type": "call_tool", "tool": "get_revenue", "args": {"quarter": quarter}, "key": f"revenue:{quarter}"}


model = ScriptedModel(
    [
        # Turn 1: fetch both figures, and try to sneak in a claim about a result not seen yet.
        {
            "actions": [
                call("Q2"),
                call("Q3"),
                {"type": "claim", "key": "q2_is_big", "value": True, "follows_from": ["revenue:Q2"]},
            ]
        },
        # Turn 2: compute growth with a formula the runtime re-executes, then answer.
        {
            "actions": [
                {
                    "type": "claim",
                    "key": "growth:Q3_vs_Q2",
                    "formula": GROWTH,
                    "claim": "Q3 revenue grew 4.65% over Q2",
                    "follows_from": ["revenue:Q2", "revenue:Q3"],
                },
                {
                    "type": "claim",
                    "key": "trend",
                    "value": "modest",
                    "claim": "Growth between 3% and 8% is modest",
                    "follows_from": ["growth:Q3_vs_Q2"],
                },
                {
                    "type": "answer",
                    "text": "Q3 revenue grew 4.65% over Q2, which is modest growth.",
                    "follows_from": ["growth:Q3_vs_Q2", "trend"],
                },
            ]
        },
    ],
    name="scripted-llm",
)

agent = Agent(model, tools=[get_revenue], dependencies="declared")
report = agent.run("Compare Q2 and Q3 revenue and assess the growth trend.")

print("ANSWER:", report.answer)
print("\nREJECTED:", *report.rejections, sep="\n  ")
print("\nPROOF:")
print(report.proof.render(ascii=True))
print()
print(report.verify())

print("\n--- Q2 revenue is restated to $4.1B ---\n")
agent.kb.correct("revenue:Q2", 4.1e9, reason="restated in 10-K/A")
print("Answer is stale:", report.stale)

# What the model would say when asked to re-derive each belief from the current inputs.
model.add(
    {
        "actions": [
            {
                "type": "claim",
                "key": "growth:Q3_vs_Q2",
                "formula": GROWTH,
                "claim": "Q3 revenue grew 9.76% over Q2",
                "follows_from": ["revenue:Q2", "revenue:Q3"],
            }
        ]
    },
    {
        "actions": [
            {
                "type": "claim",
                "key": "trend",
                "value": "strong",
                "claim": "Growth above 8% is strong",
                "follows_from": ["growth:Q3_vs_Q2"],
            }
        ]
    },
    {
        "actions": [
            {
                "type": "claim",
                "key": "answer",
                "follows_from": ["growth:Q3_vs_Q2", "trend"],
                "value": "Q3 revenue grew 9.76% over Q2, which is strong growth.",
            }
        ]
    },
)
for change in agent.repair():
    print(change)

print("\nREPAIRED ANSWER:", report.answer)
print("Verified:", report.verify().ok)
