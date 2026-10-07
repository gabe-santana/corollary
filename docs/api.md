# API reference

Everything listed here is importable from the package root (`from corollary import ...`) unless a module
is given. Signatures show keyword-only arguments after `*`. Every class and function also has a docstring
(`help(corollary.BeliefBase.assert_)`).

## Kernel

### `BeliefBase(*, trust=None, clock=None, rules=(), constraints=(), ledger=None)`

Every `key_or_ref` parameter below, `proof()`'s `*keys_or_refs`, and `derive()`'s `*inputs` and
`justify()`'s `antecedents=` also accept a `Belief` directly (meaning its exact revision), so a value
returned by `assert_()`, `derive()` or `justify()` can be passed straight back in.

**Premises**

| Method | Description |
|---|---|
| `assert_(key, value, *, source="human:user", claim="", confidence=None, valid_until=None, ttl=None, half_life=None, origin=None, supersede=False, metadata=None) -> Belief` | Assert a premise. See [asserting](guides/belief-base.md#asserting-premises) and [confidence](guides/confidence.md). |
| `assume(key, value, *, by="user", **kwargs) -> Belief` | Assert with an `assumption` source. |
| `add_document(name, text)` | Register a document for citations. |
| `documents -> Mapping[str, str]` | Registered documents (a copy). |
| `cite(key, value, *, document, quote, claim="", confidence=None, check_value=True, **kwargs) -> Belief` | Assert a premise grounded in a span-checked quote. |

**Conclusions**

| Method | Description |
|---|---|
| `register_rule(rule, *, replace=False) -> Rule` | Register a rule for derivation, re-derivation and replay; `replace=True` swaps a same-named rule. |
| `rules -> Mapping[str, Rule]` | Registered rules. |
| `derive(key, rule, *inputs, unless=(), claim="", metadata=None) -> Belief` | Derive with a rule (a `Rule`, callable, or rule name). |
| `justify(key, value, *, antecedents, source, claim="", formula=None, confidence=None, unless=(), inputs=None, note="", metadata=None) -> Belief` | Record a conclusion with explicit antecedents; formulas must reproduce `value`. |

**Change**

| Method | Description |
|---|---|
| `retract(key_or_ref, *, reason="", fault="none") -> list[Belief]` | Retract every `IN` revision of a key, or one ref. `fault="source"` records the accountable sources as wrong. |
| `restore(key_or_ref) -> Belief` | Undo a retraction. |
| `correct(key_or_belief, value, *, source=None, reason="", fault="none", **assert_kwargs) -> Belief` | Retract the current revision and assert the new value in one call. `source` defaults to the corrected revision's own source. |
| `propagate(*, rederive=None, include_kept=False, max_rounds=100) -> Propagation` | Re-derive what lost support; return the diff. |
| `changes(*, include_kept=False) -> list[Change]` | Net status changes since the log was last read (clears it). |
| `refresh() -> list[Belief]` | Apply expiries; return beliefs that just expired. |
| `stale() -> list[Belief]` | Latest revisions that are `OUT` only because their evidence expired. |
| `now() -> datetime` | The base's clock. |

**Conflicts**

| Method | Description |
|---|---|
| `add_constraint(constraint_or_name, keys=None, predicate=None, *, description="") -> Constraint` | Register an invariant. |
| `conflicts() -> list[Conflict]` | Open value and constraint conflicts. |
| `conflicted_keys() -> set[str]` | Keys in a value conflict. |
| `resolve(conflict, *, keep=None, retract=None, reason="", learn=False) -> Resolution` | Resolve by hand; `learn=True` teaches the trust ledger. |
| `resolve_conflicts(resolver, *, max_rounds=10) -> list[Resolution]` | Apply a policy until no more conflicts can be resolved. |

**Queries**

| Method | Description |
|---|---|
| `status(key_or_ref) -> Status` | `IN` / `OUT`. |
| `get(key, default=None)` | Believed `Belief`, or `default`. |
| `kb[key] -> Belief`, `value(key)` | Believed revision / its value; raise if none or conflicted. |
| `key_or_ref in kb` | Whether it is believed. |
| `latest(key)`, `revisions(key)` | Most recent revision; all revisions. |
| `keys(status=Status.IN)`, `beliefs(status=Status.IN)` | Listings; `None` for all. |
| `len(kb)`, `iter(kb)` | Count / iterate `IN` revisions. |
| `confidence(key_or_ref) -> float` | Effective confidence; see [Confidence](guides/confidence.md). |
| `reliability(source) -> float` | A source's learned reliability, from the trust policy's prior. |
| `record_outcome(source, correct, *, reason="")` | Record external ground truth in the trust ledger. |
| `faded(threshold=None) -> list[Belief]` | Believed premises whose decaying confidence fell below a threshold. |
| `valid_until(key_or_ref) -> datetime \| None` | Earliest expiry along the support. |
| `support(key_or_ref) -> Justification \| None` | Current supporting justification. |
| `justifications(key_or_ref) -> list[Justification]` | All justifications of a revision. |
| `dependents(key_or_ref, *, transitive=True) -> list[Belief]` | Downstream beliefs. |
| `why_out(key_or_ref) -> str \| None` | Why a belief is `OUT`. |
| `explain(key_or_ref) -> str` | Readable summary. |
| `proof(*keys_or_refs) -> Proof` | Snapshot of the support graph. |
| `history -> list[Event]`, `transcript() -> str` | Audit trail. |
| `trust: TrustPolicy` | The trust policy (assignable). |
| `ledger: TrustLedger` | The trust ledger this base learns in. |

**Persistence**

| Method | Description |
|---|---|
| `to_dict()`, `BeliefBase.from_dict(data, *, rules=(), constraints=(), trust=None, clock=None)` | In-memory snapshots. |
| `save(path)`, `BeliefBase.load(path, **kwargs)` | JSON files. |

### `Rederivation(belief, justification, inputs, missing)` and `Derived(value, claim="", formula=None, confidence=None, antecedents=None)`

The request a re-deriver receives and the result it returns from `propagate(rederive=...)`. Return `None`
to leave the belief pending.

### `Event(at, action, ref, detail)`

An entry of `kb.history`.

### `TrustLedger(*, prior_weight=10.0, memory_half_life=None, max_history=10_000)`

Learns source reliability from outcomes: `(prior × prior_weight + correct) / (prior_weight + total)`, with
older outcomes down-weighted when `memory_half_life` is set. Share one across belief bases with
`BeliefBase(ledger=...)`.

| Method | Description |
|---|---|
| `record(source, correct, *, at=None, reason="")` | Record an outcome. |
| `reliability(source, prior, *, at=None) -> float` | Estimated reliability, starting from `prior`. |
| `weighted(source, *, at=None) -> (correct, total)` | Decay-weighted counts. |
| `record_of(source, *, at=None) -> SourceRecord` | Counts, weighted counts and the last outcome. |
| `outcomes(source) -> list[Outcome]`, `sources() -> list[str]` | Raw history. |
| `reset(source=None)` | Forget one source, or all. |
| `to_dict()`, `from_dict()`, `save(path)`, `load(path)` | Persistence. |
| `version: int` | Incremented on every change. |

`Outcome(at, correct, reason)` and `SourceRecord(source, correct, wrong, weighted_correct, weighted_total,
last_outcome)` are the records it returns. Sources are tracked as `kind:name`, without call arguments.

## Data types

### `Belief(key, value, source, revision=1, claim="", confidence=1.0, created_at=..., metadata={})`

Immutable. Properties: `ref` (`key@revision`), `text` (claim or `key = value`). `to_dict()` / `from_dict()`.

### `Source(kind, name, detail={})`

Constructors: `Source.tool(name, args=None, *, origin=None)`, `Source.document(name, quote=None, *, origin=None)`,
`Source.human(name="user")`, `Source.model(name)`, `Source.rule(name)`, `Source.assumption(name="user")`,
`Source.parse("kind:name")`. Properties: `grounded`, `id` (`kind:name`), `origin` (independence group,
defaults to `id`), `args`, `quote`. Method: `with_origin(origin)`.

### `SourceKind`

`TOOL`, `DOCUMENT`, `HUMAN`, `MODEL`, `RULE`, `ASSUMPTION`.

### `Status`

`IN`, `OUT`.

### `Justification`

Fields: `id`, `conclusion`, `kind` (`JustificationKind.PREMISE | RULE | MODEL`), `antecedents`, `unless`,
`inputs`, `source`, `rule`, `formula`, `confidence`, `valid_until`, `half_life`, `note`, `created_at`.
`confidence` is the prior in the source for a premise, and the step's certainty otherwise. Properties:
`is_premise`, `rederivable`. Methods: `expired(now)`, `freshness(now)`, `describe()`.

### `Change(kind, belief, reason)`, `ChangeKind`, `Pending(belief, reason)`

`ChangeKind` is `IN`, `OUT` or `KEPT`. `Change.key` and `Change.ref` are shortcuts.

### `Propagation`

A sequence of `Change`s with extra fields: `pending`, `conflicts`, `rederived`. Properties: `retracted`,
`added`, `kept`, and `settled` (nothing pending, no open conflict). Method: `of_kind(kind)`. Like any
sequence it is falsy when there are no changes, even with pending beliefs or open conflicts.

## Rules and tools

### `rule(fn=None, /, *, name=None, confidence=1.0, description=None) -> Rule`

Decorator. `Rule(name, fn, confidence=1.0, description="")` is callable.

### `tool(fn=None, /, *, name=None, trust="high", ttl=None, half_life=None, origin=None, description=None) -> Tool`

Decorator. `Tool` is callable and has `bind(args)`, `default_key(args)`, `parameters_schema()`,
`describe()`.

### `TrustPolicy(sources=..., tool_levels=..., overrides={}, min_confidence=0.0, source_rank=..., corroboration=True)`

Methods: `confidence_for(source)`, `tool_confidence(trust)`, `rank(source)`,
`TrustPolicy.from_mapping(mapping)`.

## Conflicts

### `Conflict(id, kind, subject, beliefs, description, proofs)`

`kind` is `ConflictKind.VALUE` or `CONSTRAINT`. Properties: `keys`, `refs`. Method: `explain()`.

### `Constraint(name, keys, predicate, description="")` and `Resolution(conflict_id, retract, reason, authoritative=False)`

An `authoritative` resolution is treated as ground truth and teaches the trust ledger.

### Resolvers

`PreferHigherConfidence(margin=0.0)`, `PreferNewest()`, `PreferSource(order=None)`, `AskHuman(ask, *, learn=True)`. Any
callable `(conflict, kb) -> Resolution | None` is a resolver.

## Proofs and verification

### `Proof(roots, steps, created_at)`

Build with `kb.proof(...)` or `Proof.build(kb, *keys)`. Access: iteration, `len`, `in`, `step(key_or_ref)`,
`by_ref`, `premises`, `derived`, `valid`. Output: `render(*, show_sources=True, ascii=False)`, `str()`,
`to_mermaid()`, `to_dot()`, `to_dict()`, `to_json()`, `from_dict()`, `from_json()`. Comparison:
`diff(other) -> ProofDiff(added, removed, changed, status_changed)`. Verification: `verify(kb=None, *, checks=None, at=None)`.

### `ProofStep(belief, status, confidence, justification)`

Properties: `ref`, `antecedents`, `is_premise`.

### `Verifier(checks=None)`

`verify(proof, *, kb=None, at=None) -> VerificationReport`. With no checks, it runs all built-in checks.

### `VerificationReport(results)`

`ok`, `bool()`, `errors`, `warnings`, `raise_for_errors()`, `str()`.

### `CheckResult(check, passed, message, ref=None, severity=Severity.ERROR)` and `Severity`

### `corollary.verify`

`Check` (protocol), `VerificationContext(at, kb, documents, rules)`, `StructureCheck`, `GroundingCheck`,
`ArithmeticCheck(rel_tol=1e-6)`, `CitationCheck`, `TemporalCheck`,
`NumericProvenanceCheck(ignore_below=13, ignore_years=True)`, `DEFAULT_CHECKS`.

## Agents

### `Agent(model, beliefs=None, tools=(), *, documents=None, rules=(), trust=None, projector=None, dependencies="conservative", resolver=None, max_steps=12, repair_attempts=2, self_consistency=1, learn_from_checks=True, instructions="", system_prompt=SYSTEM_PROMPT)`

| Method / attribute | Description |
|---|---|
| `run(task, *, max_steps=None, instructions="") -> Report` | Work on a task until answered or out of steps; `report.error` holds a model failure. |
| `repair(*, include_kept=False) -> Propagation` | Re-derive everything that lost support. |
| `reverify(*, include_kept=False) -> Propagation` | Re-run expired or faded tool calls, then repair. |
| `narrow(key) -> NarrowResult` | Prune unnecessary dependencies by ablation. |
| `kb` / `beliefs`, `model`, `tools`, `projector`, `dependencies`, `resolver` | Configuration. |

### `Report`

Fields: `task`, `key`, `kb`, `completed`, `steps`, `changes`. Properties: `belief`, `answer`, `stale`,
`proof`, `rejections`. Method: `verify(checks=None, *, raise_on_error=False)`.

### `StepRecord(index, prompt, response, accepted, rejected)`, `NarrowResult(key, kept, pruned, calls)`

### `Dependencies`

`CONSERVATIVE`, `DECLARED`.

### `Projector(*, max_beliefs=500, min_confidence=None, max_document_chars=50_000, show_sources=True)`

`select(kb, scope=None) -> list[Belief]`, `project(kb, *, task, tools=(), feedback=(), scope=None,
instructions="", include_documents=True) -> Projection`. `Projection(text, visible)` has `refs`, `keys`,
`belief(key)`.

## The contract

`SYSTEM_PROMPT`, `parse_response(text) -> ParsedResponse(actions, errors)`, and the action types
`ToolCall(tool, args, key, claim)`, `Cite(document, quote, key, value, claim)`,
`Claim(key, value, claim, follows_from, formula, confidence)`, `Answer(text, follows_from, confidence)`.
`Action` is the union of those four action types and is the type of each item in `ParsedResponse.actions`.
`corollary.contract` also provides `CONTRACT_SCHEMA` and `extract_json`.

`corollary.formula` provides `evaluate(formula, values)`, `formula_keys(formula)` and `FUNCTIONS`.

## Models

`Model` (protocol: `name`, `complete(system, prompt)`), `AnthropicModel`, `OpenAIModel`, `CallableModel`,
`ScriptedModel`. `corollary.models.resolve_model(spec)` turns a string into a model. See
[Models](guides/models.md).

## Errors

All derive from `CorollaryError`.

| Error | Raised when |
|---|---|
| `InvalidKeyError` (`ValueError`) | A key is empty or contains whitespace, `@` or braces |
| `UnknownBeliefError` (`KeyError`) | A key or ref doesn't exist |
| `NotBelievedError` (`LookupError`) | A belief is `OUT` where an `IN` one is required |
| `UnresolvedConflictError` (`LookupError`) | A key has incompatible believed values |
| `CircularDefeatError` | A justification would make a belief depend on its own absence |
| `CitationError` (`ValueError`) | A quote or value isn't in the document |
| `FormulaError` (`ValueError`) | A formula is invalid, forbidden, or doesn't reproduce a value |
| `RuleError` | A rule is unknown, conflicting, or raised |
| `ContractViolation` | A model response breaks the contract (`.errors` lists each problem) |
| `ModelError`, `ModelRefusalError` | A model call failed or was declined |
| `VerificationError` | `raise_for_errors()` on a failing report |
