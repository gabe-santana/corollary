# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). Until 1.0, minor versions may contain breaking changes.

## [Unreleased]

### Added

- `retract`, `restore`, `status`, `confidence`, `valid_until`, `support`, `justifications`,
  `dependents`, `why_out`, `explain` and `proof`, plus `derive()`'s `*inputs` and `justify()`'s
  `antecedents=`, now accept a `Belief` directly (meaning its exact revision) wherever a key or ref
  was accepted, so a value returned by `assert_()`, `derive()` or `justify()` can be passed straight
  back in instead of stringified to `.ref` first. Anything else raises `TypeError`.

### Changed

- The public `Action` type is now exported and documented for annotating parsed responses.
- Documented verification, formula and contract submodules now declare their supported public exports.
- `Proof.to_json()` serialization errors now identify the first belief whose value JSON cannot represent.
- `Report` and `StepRecord` now have a short `repr` that summarizes (task, completion, step/rejection
  counts, error) instead of dumping every step's full prompt and raw model response. `str(report)` is
  unchanged.
- `Report.__repr__`'s `error` is now truncated too (shown as `type(message)`, message truncated to ~60
  chars), so a model error with a multi-kilobyte message no longer blows the repr back up. `StepRecord`'s
  repr also shows `response_length` next to `prompt_length`.

## [0.1.0a4] - 2026-10-03

### Changed

- `Proof.to_json()` raises `TypeError` for a value JSON can't represent (a `Decimal`, a `date`) instead of
  silently writing its string form, which changed the value's type on reload. Pass `default=str` to keep
  the old behavior. Thanks to @Jah-yee for the first outside contribution (#24).

## [0.1.0a3] - 2026-10-03

### Changed

- `refresh()`, which `derive()`, `justify()`, the projector and the agent call, now skips its scan until the
  earliest validity window can have ended. With 3,000 expiring premises, 3,000 derivations go from 3.96 s
  to 0.05 s.

## [0.1.0a2] - 2026-10-02

### Added

- `Agent(instructions=...)` and `run(task, instructions=...)` for domain guidance, without replacing the
  contract's system prompt.
- `Report.error`: a model failure (refusal, truncation, API error) ends the run and is reported there,
  instead of raising and losing the run's steps.
- `async def` tools, and type checking with unambiguous coercion of tool arguments (`"3"` for an `int`).
- Logging to the `corollary` logger: model failures, tool exceptions with their traceback, failed
  re-verifications, and each step at `DEBUG`.
- `Propagation.settled`, `ProofDiff.status_changed`, and `register_rule(..., replace=True)`.

### Changed

- Citation and provenance checks match a number at the scale its unit states: "$4 million" no longer
  matches 4.3e9, and "4 percent" no longer matches 4.1e9. A value stored in a smaller unit still matches
  (4300, in millions, for "$4.3 billion"), and a bare number may still be any magnitude or a percentage.
- String and boolean values in citations must appear as a whole word or phrase ("false" is not in
  "falsehood"); a numeric string is matched as a number.
- The provenance check no longer counts a model claim's own value as support (unless a formula computed
  it), and ignores dates, times and ordinals.
- `TrustPolicy(sources=...)` and `tool_levels=` are merged with the defaults instead of replacing them.
- `resolve(conflict, keep="k")` asks for a ref when the key is on several sides of the conflict.
- Integers compare exactly; NaN equals NaN; lists, tuples and dicts compare element by element.
- `derive()` and `justify()` refuse inputs whose evidence has expired.
- A source re-asserting a value it already gave replaces its justification instead of adding one, and is
  not counted as a confirmation.
- `BeliefBase.save()` and `TrustLedger.save()` write atomically. Loading a damaged snapshot raises
  `ValueError`; loading one that uses unregistered rules warns.
- `AnthropicModel` leaves out server-side fallbacks for Bedrock, Vertex AI and Foundry clients.

### Fixed

- A tool call or citation could take the reserved answer key.
- A re-derivation whose response failed to parse gave up instead of retrying.
- JSON extraction picked the first JSON value in a response, even a stray list before the actions.
- Formulas could crash the agent (`OverflowError`, `RecursionError`) or hang on chained powers; every
  intermediate is now a finite float.
- A failed assertion (for example a naive `valid_until`) left a half-built node that broke save and load.
- `supersede=True` did nothing when the value already existed; a lone string passed as a key list was
  split into characters; retracting a retracted ref penalized its source again; a resolver naming
  unknown refs crashed with `KeyError`.
- Confidence ignored in-place trust policy edits and a forgetting ledger's decay, and took quadratic time
  on long chains. Re-derivation ordering was O(n * depth).
- `Proof.render()` hit the recursion limit on deep proofs; DOT and Mermaid exports mis-escaped quotes and
  backslashes; `Proof.from_dict()` accepted any format.
- `verify(at=...)` crashed on a naive datetime.
- `OpenAIModel` missed refusals, crashed on an empty `choices` list, and overrode an explicit
  `response_format`.
- `run(max_steps=0)` used the default budget; a generator `scope` dropped conflicts from the context.

## [0.1.0a1] - 2026-10-02

First public pre-release.

### Added

- **Kernel.** `BeliefBase`: a justification-based truth maintenance system over versioned beliefs.
  Incremental labeling restricted to the affected region, well-founded support (cycles alone never keep a
  belief `IN`), non-monotonic `unless` justifications with odd-loop detection and rollback, retraction and
  restoration, and a change log with reasons for every transition.
- **Re-derivation.** `propagate()` re-runs rules automatically and model-derived beliefs through a
  re-deriver, with early cutoff when a recomputed value is unchanged.
- **Rules.** The `@rule` decorator for deterministic derivations, replayable by the verifier.
- **Conflicts.** Value conflicts and user-defined `Constraint`s are first-class `Conflict` objects
  carrying both support chains. `resolve()`, plus the resolvers `PreferHigherConfidence`, `PreferNewest`,
  `PreferSource` and `AskHuman`.
- **Time.** Validity windows (`ttl`, `valid_until`) on evidence, `refresh()`, `stale()`, and in-place
  renewal that avoids needless cascades.
- **Documents.** `add_document()` and `cite()` with span-checked quotes and value checks.
- **Proofs.** `Proof` snapshots with tree rendering, Mermaid and Graphviz export, JSON round-trip and diffs.
- **Verification.** `Verifier` with structure, grounding, arithmetic (formula and rule replay), citation,
  temporal and numeric-provenance checks, plus a `Check` protocol for custom checks.
- **Agent runtime.** `Agent` with the claim contract, runtime-executed tools that emit premises, a
  projector that builds every context from `IN` beliefs only, conservative and declared dependency
  policies, `repair()`, `reverify()` and ablation-based `narrow()`.
- **Models.** `AnthropicModel` (Claude, via the official SDK), `OpenAIModel` (any Chat Completions
  endpoint), `CallableModel` and `ScriptedModel`.
- **Learned reliability.** `TrustLedger` records when sources turn out right or wrong and re-estimates
  their reliability from the trust policy's prior, optionally forgetting old outcomes
  (`memory_half_life`). Ledgers can be shared across belief bases and are saved with a base that owns one.
- **Outcomes.** `retract(..., fault="source")`, `resolve(..., learn=True)`, `AskHuman` decisions,
  independent confirmations, `record_outcome()`, and the agent's verified formulas and citations
  (`learn_from_checks`) all feed the ledger.
- **Corroboration.** Agreeing premises from independent origins combine by noisy-OR. Sources declare a
  shared origin with `origin=` (on `assert_`, `Source` and `@tool`); `TrustPolicy(corroboration=False)`
  turns it off.
- **Evidence decay.** `half_life=` on `assert_` and `@tool` fades confidence without changing status.
  `faded()` lists faded premises, and `Agent.reverify()` now refreshes them too.
- **Self-consistency.** `Agent(self_consistency=k)` samples unverified claims `k` times and uses the
  agreement rate as the step's certainty.
- A confidence guide in the documentation.
- **Persistence.** JSON snapshots with `save()` / `load()`, including the trust ledger.
- Examples: a 20-conclusion self-repairing report, an offline agent, and a Claude agent.

[Unreleased]: https://github.com/gabe-santana/corollary/compare/v0.1.0a4...HEAD
[0.1.0a4]: https://github.com/gabe-santana/corollary/compare/v0.1.0a3...v0.1.0a4
[0.1.0a3]: https://github.com/gabe-santana/corollary/compare/v0.1.0a2...v0.1.0a3
[0.1.0a2]: https://github.com/gabe-santana/corollary/compare/v0.1.0a1...v0.1.0a2
[0.1.0a1]: https://github.com/gabe-santana/corollary/releases/tag/v0.1.0a1
