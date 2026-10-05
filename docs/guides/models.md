# Models

In Corollary a model is a stateless proposer. It receives one prompt built by the projector and returns
one response in the [claim contract](contract.md). It never sees earlier turns, so an adapter only needs a
single-shot completion:

```python
class Model(Protocol):
    @property
    def name(self) -> str: ...  # recorded as the source of its claims
    def complete(self, system: str, prompt: str) -> str: ...
```

The agent accepts a model instance, or a string:

```python
Agent("anthropic:claude-opus-5-5", ...)
Agent("claude-sonnet-5-5", ...)  # bare claude-* ids select the Anthropic adapter
Agent("openai:<model-id>", ...)
```

## Claude

```bash
pip install "corollary[anthropic]"
```

```python
from corollary import AnthropicModel

model = AnthropicModel(
    "claude-opus-5-5",  # default
    max_tokens=16_000,
    effort="high",  # output_config.effort: low, medium, high, xhigh, max
    structured=False,  # constrain output to the contract's JSON schema
    fallbacks="default",  # server-side refusal fallback
    client=None,  # a configured anthropic.Anthropic(), or a platform client
)
```

- **Credentials** are resolved by the SDK: `ANTHROPIC_API_KEY`, or a profile from `ant auth login`.
- **Effort** defaults to `"high"`, because every call carries reasoning the runtime will hold the model
  to. Lower it for cheaper, simpler tasks.
- **Fallbacks.** With `fallbacks="default"`, the request uses the server-side refusal fallback (beta
  `server-side-fallback-2026-07-01`), so a declined request can be served by a fallback model in the same
  call. This is available on the Claude API only, so it is left out automatically when `client` is an
  Amazon Bedrock, Google Vertex AI or Microsoft Foundry client.
- **Refusals** that still happen raise `ModelRefusalError`, and truncated responses raise `ModelError`.
  `Agent.run()` catches both and returns its report with the error in `report.error`.
- **Other platforms.** Pass a platform client, for example `anthropic.AnthropicBedrockMantle(...)`, as
  `client=`.
- `model.request(system, prompt)` returns the exact keyword arguments that would be sent, which is useful
  for logging and tests.

## OpenAI-compatible endpoints

```bash
pip install "corollary[openai]"
```

```python
import openai
from corollary import OpenAIModel

model = OpenAIModel("<model-id>")  # api.openai.com
local = OpenAIModel("llama3", client=openai.OpenAI(base_url="http://localhost:11434/v1", api_key="x"))
```

`OpenAIModel` uses the Chat Completions API, so it works with any compatible server (vLLM, Ollama,
LM Studio, ...). It requests JSON mode by default. Pass `json_mode=False` for servers that don't support
it. Extra keyword arguments (`temperature=0`, ...) are forwarded; an explicit `response_format` replaces
JSON mode. Refusals (`message.refusal`, or `finish_reason == "content_filter"`) raise `ModelRefusalError`.

[`examples/agent_ollama.py`](https://github.com/gabe-santana/corollary/blob/main/examples/agent_ollama.py)
runs the full agent loop, including a restatement and `repair()`, against a local Ollama model. Pass the
model name as the first argument or set `OLLAMA_MODEL`; set `OLLAMA_HOST` if the server isn't on
`localhost:11434`.

## Any function

```python
from corollary import CallableModel


def my_model(system: str, prompt: str) -> str: ...  # call LiteLLM, a local model, an internal gateway...


agent = Agent(CallableModel(my_model, name="internal-llm"), ...)
```

## Scripted responses, for tests and demos

```python
from corollary import ScriptedModel

model = ScriptedModel(
    [
        {"actions": [...]},  # dicts and lists are serialized to JSON
        '{"actions": []}',  # strings are returned as-is
        lambda prompt: {"actions": [...]},  # callables receive the prompt
    ]
)
model.add({"actions": [...]})  # append more responses later
model.calls  # Call(system, prompt, response) for each call
model.remaining  # responses left
```

`ScriptedModel` makes agent behavior fully deterministic, which is how Corollary's own test suite covers
the agent runtime without a network.

## Writing an adapter

An adapter should:

1. Send `system` as the system prompt and `prompt` as a single user message. There is no history to
   manage.
2. Return the text of the response. Don't parse it: the runtime does that and reports problems back to
   the model.
3. Raise `ModelError` for transport failures and truncated output, and `ModelRefusalError` for refusals.
4. Expose a stable `name`. It is recorded as the source of every claim (`model:<name>`) and used by the
   trust policy.

Contributions of adapters for more providers are welcome; see
[CONTRIBUTING.md](https://github.com/gabe-santana/corollary/blob/main/CONTRIBUTING.md).
