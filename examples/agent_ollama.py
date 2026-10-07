"""The same task with a local model served by Ollama, through its OpenAI-compatible endpoint.

    pip install "corollary[openai]"
    ollama pull llama3.1
    python examples/agent_ollama.py              # llama3.1
    python examples/agent_ollama.py qwen2.5      # or any model you have pulled

The model name can also be set with OLLAMA_MODEL, and the server address with OLLAMA_HOST
(default http://localhost:11434). Small local models follow the claim contract less reliably than
hosted ones, so expect more rejected answers; Corollary rejects them instead of storing them.
"""

from __future__ import annotations

import os
import sys

try:
    import openai
except ImportError:
    sys.exit('This example needs the OpenAI SDK: pip install "corollary[openai]"')

from corollary import Agent, OpenAIModel, tool

FILINGS = {"Q1": 4.0e9, "Q2": 4.3e9, "Q3": 4.5e9, "Q4": 4.8e9}


@tool(trust="high")
def get_revenue(quarter: str) -> float:
    """Quarterly revenue in USD from SEC filings. quarter is one of Q1, Q2, Q3, Q4."""
    return FILINGS[quarter]


def ollama_url() -> str:
    """The server address from OLLAMA_HOST, which Ollama allows without a scheme (localhost:11434)."""
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    return host if "://" in host else f"http://{host}"


def main() -> None:
    model_name = sys.argv[1] if len(sys.argv) > 1 else os.getenv("OLLAMA_MODEL", "llama3.1")
    url = ollama_url()
    client = openai.OpenAI(base_url=f"{url}/v1", api_key="ollama")  # Ollama ignores the key, the SDK requires one
    agent = Agent(OpenAIModel(model_name, client=client), tools=[get_revenue])
    report = agent.run("Compare Q2 and Q3 revenue and assess the growth trend.")

    print("ANSWER:", report.answer)
    for rejection in report.rejections:
        print("  rejected:", rejection)
    if not report.completed:
        print("\nThe agent did not answer:", report.error or f"no answer within {agent.max_steps} steps")
        if report.error is not None:
            print(f"Is Ollama running at {url}? Start it with `ollama serve` and `ollama pull {model_name}`.")
        return
    print("\nPROOF:\n" + report.proof.render())
    print("\n" + str(report.verify()))

    print("\n--- Q2 revenue is restated to $4.1B ---\n")
    agent.kb.retract("revenue:Q2", reason="restated in 10-K/A")
    agent.kb.assert_("revenue:Q2", 4.1e9, source="tool:get_revenue")
    for change in agent.repair():
        print(change)
    print("\nREPAIRED ANSWER:", report.answer)
    print(report.verify())


if __name__ == "__main__":
    main()
