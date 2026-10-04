# The belief base

`BeliefBase` is the kernel: pure, deterministic code with no model involved. You can use it on its own for
any computation where inputs change and you want conclusions to stay consistent with them, and every
agent is built on top of one.

```python
from corollary import BeliefBase

kb = BeliefBase()
```

Constructor options: `trust=` a [`TrustPolicy`](#trust-policy), `clock=` a zero-argument callable
returning a timezone-aware `datetime` (inject one for deterministic tests), `rules=` and `constraints=`
to register up front, and `ledger=` a [`TrustLedger`](confidence.md#2-learned-reliability-the-trust-ledger)
to share across bases.

## Asserting premises

```python
kb.assert_("revenue:Q2", 4.3e9, source="tool:sec_filings", claim="Q2 revenue")
kb.assert_("ceo", "Dana Reyes", source="human:alice")
kb.assume("market_size", 2e10)  # a working hypothesis (source "assumption:user")
```

`source` is a `Source` or a `"kind:name"` string. Kinds are `tool`, `document`, `human`, `assumption`
and `model` (see [Core concepts](../concepts.md#premises-and-conclusions)). `confidence` defaults to the
trust policy's value for the source.

What happens depends on what the base already believes about the key:

| Situation | Result |
|---|---|
| New key | Revision 1 is created and is `IN`. |
| Same value already `IN` | The new source is added as a corroborating justification. No new revision. |
| Latest revision expired, same value | The evidence is renewed in place. Nothing downstream re-derives. |
| Different value, and an older revision is `IN` | A new revision is created, and the two form a **value conflict**. |
| Different value, `supersede=True` | Older `IN` revisions are retracted first, then the new one is created. |

Use `supersede=True` for a fresh observation of a moving value (a new price quote). Leave it off when two
independent sources might disagree: you want to see that conflict.

## Rules

A rule is a named, deterministic function. Decorate it with `@rule`:

```python
from corollary import rule


@rule
def growth(previous: float, current: float) -> float:
    """Percent growth from previous to current."""
    return (current - previous) / previous * 100


@rule(name="margin", confidence=0.99)
def gross_margin(revenue: float, cogs: float) -> float:
    return (revenue - cogs) / revenue * 100
```

The name is what gets stored in the graph, so it must be stable if you [persist](persistence.md) the base.
Plain functions and lambdas also work with `derive`. Lambdas get generated names (`lambda:1`, ...),
which is fine for scripts but not for anything you will save and reload.

## Deriving conclusions

```python
kb.derive("growth:Q3_vs_Q2", growth, "revenue:Q2", "revenue:Q3")
```

Inputs are passed positionally, in the order the rule expects. Each input is a key (its currently
believed value) or a ref (that exact revision). Every input must be `IN`, otherwise `NotBelievedError` is
raised, and a conflicted input raises `UnresolvedConflictError`.

Derived beliefs remember their recipe: when an input changes, `propagate()` re-runs the rule.

### Exceptions and defaults: `unless`

A justification can hold only while some other belief is *absent*:

```python
kb.assert_("filing", "10-K", source="document:10-K")
kb.derive("figures_final", lambda f: True, "filing", unless=["restatement"])

kb.status("figures_final")  # IN
kb.assert_("restatement", True, source="document:10-K/A")
kb.status("figures_final")  # OUT, defeated by: restatement
kb.retract("restatement")
kb.status("figures_final")  # IN again
```

A belief may not depend on its own absence, directly or through a chain (`a unless b`, `b from a`). Such a
graph has no stable labeling, so the justification is rejected with `CircularDefeatError` and the base is
left exactly as it was.

### Model-style justifications: `justify`

`justify` records a conclusion with explicit antecedents. The agent uses it for model claims, and you can
use it for anything that isn't a registered rule:

```python
kb.justify(
    "growth",
    4.651162790697675,
    antecedents=["revenue:Q2", "revenue:Q3"],
    source="model:analyst",
    formula="({revenue:Q3} - {revenue:Q2}) / {revenue:Q2} * 100",
)
```

When a `formula` is given it is evaluated against the antecedent values, and the call raises
`FormulaError` unless it reproduces `value`. Formulas use `{key}` placeholders and allow only arithmetic,
`abs`, `min`, `max`, `round`, `sqrt`, `log` and `exp`. See [the contract](contract.md#formulas).

## Retracting and restoring

```python
kb.retract("revenue:Q2", reason="restated in 10-K/A")  # every IN revision of the key
kb.retract("revenue:Q2@1")  # exactly one revision
kb.restore("revenue:Q2")  # undo the latest retraction
```

Retraction takes effect immediately: every dependent goes `OUT` before `retract` returns. Nothing is
deleted. Retracted revisions remain queryable with their reason.

Say whose mistake it was with `fault`, so the trust ledger can learn from it:

```python
kb.retract("revenue:Q2", reason="restated in 10-K/A")  # fault="none": the world changed
kb.retract("price:ACME", reason="scraper returned stale data", fault="source")  # the source was wrong
```

With `fault="source"`, every source accountable for the belief is recorded as wrong, and its
reliability (and the confidence of everything else it reported) drops. See [Confidence](confidence.md).

## Correcting a fact

Retracting the old revision and asserting the new one is the most common thing people do with
Corollary, so `correct()` does both in one call:

```python
kb.correct("revenue:Q2", 4.1e9, reason="restated in 10-K/A")
```

`source` defaults to the source of the revision being corrected, so the call above needs no
`source=` of its own. Pass one explicitly when the correction comes from somewhere else:

```python
kb.correct("price:ACME", 41.2, source="tool:refresh", reason="stale quote", fault="source")
```

`reason` and `fault` mean the same as `retract()`'s. If the new value fails to assert (an invalid
value, source or confidence), nothing changes — `correct()` is built on `assert_(supersede=True)`'s
own rollback.

## Propagating changes

```python
kb.correct("revenue:Q2", 4.1e9, reason="restated")
result = kb.propagate()
```

`propagate()` does two things:

1. **Re-derives** every derived belief that lost support and whose inputs are available again. Rules are
   re-run directly. Model-derived beliefs need a re-deriver, which `Agent.repair()` provides. Without one,
   they are listed in `result.pending`.
2. **Reports** every net status change since the change log was last read, in dependency order.

```python
for change in result:  # Change(kind, belief, reason)
    print(change)

result.retracted  # changes of kind OUT
result.added  # changes of kind IN
result.rederived  # beliefs created or restored by re-derivation
result.pending  # Pending(belief, reason): e.g. "waiting for: revenue:Q2"
result.conflicts  # open conflicts after propagation
```

Pass `include_kept=True` to also list derived beliefs that were unaffected (`KEPT`).

**Early cutoff.** If a re-derived value equals the old one, the old revision gets a new justification and
comes back `IN` without a new revision. Its dependents come back with it, without being recomputed. In the
[flagship example](https://github.com/gabe-santana/corollary/blob/main/examples/self_repairing_report.py),
a restatement changes 12 of 20 conclusions, and the growth trend is recomputed, found unchanged, and stops
the cascade.

**The change log.** `propagate()` and `kb.changes()` both return net changes since the log was last read,
and clear it. After building a base, call `kb.changes()` once so that the next diff shows only what your
correction changed. An agent's `run()` does this for you and stores the run's changes on the report.

## Querying

```python
kb.status("growth")  # Status.IN / Status.OUT (any revision of a key, or one ref)
kb.value("growth")  # value of the believed revision; raises if OUT or conflicted
kb["growth"]  # the believed Belief; raises if none
kb.get("growth")  # the believed Belief, or None (or a default)
"growth" in kb  # is any revision believed?

kb.latest("growth")  # most recent revision, whatever its status
kb.revisions("growth")  # every revision, oldest first
kb.keys()  # keys with an IN revision; keys(Status.OUT), keys(None)
kb.beliefs()  # IN revisions; beliefs(None) for all
len(kb), list(kb)  # count / iterate IN revisions

kb.confidence("growth")  # effective confidence along the current support
kb.valid_until("growth")  # earliest expiry along the support chain
kb.support("growth")  # the Justification currently making it IN
kb.justifications("growth")  # all justifications of the revision
kb.dependents("revenue:Q2")  # everything downstream (transitive=False for direct only)
kb.why_out("growth@1")  # "lost support: revenue:Q2"
kb.proof("growth")  # a Proof snapshot; see Verification
print(kb.explain("growth"))  # a readable summary of all of the above
```

`explain` prints something like:

```text
growth:Q3_vs_Q2@1 = 4.65116  [OUT, confidence 0.00]
  why OUT:    lost support: revenue:Q2
    j4     rule     rule:growth; from revenue:Q2@1, revenue:Q3@1
  used by:    trend:Q3@1
```

The `*` marker in that listing shows the justification currently supporting the belief.

## Trust policy

A `TrustPolicy` maps sources to confidence and sets thresholds the runtime enforces:

```python
from corollary import BeliefBase, TrustPolicy

trust = TrustPolicy(
    overrides={"tool:sec_filings": 0.999, "human:alice": 1.0},
    min_confidence=0.5,  # the projector hides beliefs below this
)
kb = BeliefBase(trust=trust)
```

| Field | Default | Meaning |
|---|---|---|
| `sources` | human 0.99, tool 0.95, document 0.9, model 0.9, assumption 0.6, rule 1.0 | confidence per source kind |
| `tool_levels` | high 0.99, medium 0.9, low 0.7 | values for `@tool(trust="...")` |
| `overrides` | `{}` | exact `"kind:name"` entries, checked first |
| `min_confidence` | 0.0 | beliefs below this are not shown to the model |
| `source_rank` | human, tool, document, rule, model, assumption | order used by `PreferSource` |
| `corroboration` | `True` | combine agreeing independent sources by noisy-OR |

`TrustPolicy.from_mapping({"tool": 0.9, "human:alice": 1.0})` builds one from a flat mapping.

The values in the policy are **priors**. The base's `TrustLedger` learns each source's actual
reliability from outcomes and adjusts them over time; see [Confidence](confidence.md).

## History

```python
for event in kb.history:  # Event(at, action, ref, detail)
    print(event)
print(kb.transcript())
```

Actions are `assert`, `support`, `renew`, `derive`, `rederive`, `retract`, `restore` and `resolve`. The
history is an audit trail. The belief base, not the history, is the source of truth.
