"""The kernel: a justification-based truth maintenance system (JTMS) over versioned beliefs.

Everything in this module is deterministic code. No language model is involved in deciding what
is believed; models only *propose* beliefs and justifications, which the kernel then labels.

Labeling semantics
------------------
A belief is ``IN`` when it is not retracted and at least one of its justifications is valid. A
justification is valid when all of its antecedents are ``IN``, none of its ``unless`` keys has an
``IN`` revision, and it has not expired. Labels are the least fixpoint of these equations, so a
group of beliefs that only support each other in a cycle is ``OUT`` unless something outside the
cycle supports it (support must be *well-founded*).

When something changes, only the region downstream of the change is relabeled, so the cost of a
retraction scales with what depends on it, not with the size of the belief base.
"""

from __future__ import annotations

import contextlib
import dataclasses
import heapq
import itertools
import json
import os
import warnings
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ._io import write_atomically
from .belief import Belief, Source, SourceKind, Status, format_value, parse_ref, utcnow, validate_key, values_equal
from .changes import Change, ChangeKind, Pending, Propagation
from .conflict import Conflict, ConflictKind, Constraint, Resolution, Resolver, describe_values
from .errors import (
    CircularDefeatError,
    CitationError,
    FormulaError,
    NotBelievedError,
    RuleError,
    UnknownBeliefError,
    UnresolvedConflictError,
)
from .formula import evaluate, formula_keys
from .justification import Justification, JustificationKind
from .ledger import TrustLedger
from .proof import Proof
from .rules import Rule
from .textmatch import contains_quote, value_in_text
from .trust import TrustPolicy

_FORMAT = "corollary.beliefbase"
_FORMAT_VERSION = 2


@dataclass(frozen=True)
class Event:
    """An entry in the belief base history. The history is a *view*, never the source of truth."""

    at: datetime
    action: str
    ref: str
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.at.isoformat(timespec='seconds')}  {self.action:<8} {self.ref}  {self.detail}".rstrip()


@dataclass(frozen=True)
class Rederivation:
    """Request passed to a re-deriver callback for a model-derived belief that went ``OUT``.

    ``inputs`` maps each recipe key that currently has a believed value to that belief;
    ``missing`` lists recipe keys that have none.
    """

    belief: Belief
    justification: Justification
    inputs: Mapping[str, Belief]
    missing: tuple[str, ...] = ()


@dataclass(frozen=True)
class Derived:
    """Result returned by a re-deriver.

    ``antecedents`` may narrow the dependencies to a subset of the request's input keys; when
    omitted, the new justification depends on every available input (the conservative choice).
    """

    value: Any
    claim: str = ""
    formula: str | None = None
    confidence: float | None = None
    antecedents: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.confidence is not None:
            _check_confidence(self.confidence)


Rederiver = Callable[[Rederivation], "Derived | None"]


@dataclass(eq=False)
class _Node:
    belief: Belief
    justifications: list[Justification] = field(default_factory=list)
    status: Status = Status.OUT
    support: Justification | None = None
    retracted: bool = False
    retract_reason: str = ""
    consumers: set[str] = field(default_factory=set)

    @property
    def ref(self) -> str:
        return self.belief.ref

    @property
    def derived(self) -> bool:
        return any(not j.is_premise for j in self.justifications)


class BeliefBase:
    """A versioned belief graph with truth maintenance.

    Args:
        trust: Confidence assigned to sources; see :class:`TrustPolicy`.
        clock: Zero-argument callable returning the current time. Inject one to make validity
            windows deterministic in tests.
        rules: Rules to register up front (also needed when loading a saved base).
        constraints: Invariants that raise a :class:`Conflict` when violated.
        ledger: A :class:`TrustLedger` to learn source reliability in. Pass the same ledger to
            many belief bases to learn across all of them; by default each base has its own.

    Basic usage::

        kb = BeliefBase()
        kb.assert_("revenue:Q2", 4.3e9, source="tool:sec_filings")
        kb.assert_("revenue:Q3", 4.5e9, source="tool:sec_filings")
        kb.derive("growth", growth_rule, "revenue:Q2", "revenue:Q3")

        kb.retract("revenue:Q2", reason="restated")
        kb.assert_("revenue:Q2", 4.1e9, source="tool:sec_filings")
        for change in kb.propagate():
            print(change)
    """

    def __init__(
        self,
        *,
        trust: TrustPolicy | None = None,
        clock: Callable[[], datetime] | None = None,
        rules: Iterable[Rule] = (),
        constraints: Iterable[Constraint] = (),
        ledger: TrustLedger | None = None,
    ) -> None:
        self.trust = trust or TrustPolicy()
        self._clock = clock or utcnow
        self.ledger = ledger if ledger is not None else TrustLedger()
        self._owns_ledger = ledger is None
        # Confidence cache, invalidated by any graph change, ledger change or (with decay) time.
        self._version = 0
        self._decaying = False
        self._conf_cache: dict[str, float] = {}
        self._conf_epoch: tuple[Any, ...] | None = None
        self._nodes: dict[str, _Node] = {}
        self._revisions: dict[str, list[str]] = {}
        self._justifications: dict[str, Justification] = {}
        self._unless_consumers: defaultdict[str, set[str]] = defaultdict(set)
        self._expiring: set[str] = set()
        # No evidence can expire before this time, so refresh() has nothing to scan until then. It is a
        # lower bound: lowered whenever evidence with an earlier window is linked, recomputed on scans.
        self._next_expiry: datetime | None = None
        self._rules: dict[str, Rule] = {}
        self._constraints: dict[str, Constraint] = {}
        self._documents: dict[str, str] = {}
        self._history: list[Event] = []
        self._ids = itertools.count(1)
        # Change tracking between two calls to propagate(): status before the first change.
        self._baseline: dict[str, Status | None] = {}
        self._reasons: dict[str, str] = {}
        self._hints: dict[str, str] = {}
        for r in rules:
            self.register_rule(r)
        for c in constraints:
            self.add_constraint(c)

    # ======================================================================================
    # Time
    # ======================================================================================

    def now(self) -> datetime:
        """Current time according to this base's clock."""
        return self._clock()

    # ======================================================================================
    # Asserting premises
    # ======================================================================================

    def assert_(
        self,
        key: str,
        value: Any,
        *,
        source: Source | str = "human:user",
        claim: str = "",
        confidence: float | None = None,
        valid_until: datetime | None = None,
        ttl: timedelta | None = None,
        half_life: timedelta | None = None,
        origin: str | None = None,
        supersede: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> Belief:
        """Assert a premise: a belief grounded directly in a source.

        * If ``key`` already has an ``IN`` revision with the same value, the new source is added
          as a corroborating justification and that revision is returned.
        * If the latest revision is ``OUT`` only because its evidence expired and the value is the
          same, the evidence is renewed in place, so nothing downstream has to be re-derived.
        * Otherwise a new revision is created. If an older revision with a different value is
          still ``IN``, the two now form a value :class:`Conflict`, unless ``supersede=True``, in
          which case the older revisions are retracted first.

        ``ttl`` (or ``valid_until``) bounds how long this piece of evidence is valid: after it, the
        evidence expires and the belief goes ``OUT``. ``half_life`` instead makes its confidence
        fade gradually (halving every ``half_life``) without changing its status.

        ``origin`` declares an independence group: sources with the same origin (two tools reading
        the same database) count once when they agree. Corroboration by a source of a *different*
        origin is recorded in the trust ledger as a success for both.
        """
        validate_key(key)
        src = Source.parse(source).with_origin(origin)
        if half_life is not None and half_life.total_seconds() <= 0:
            raise ValueError("half_life must be positive")
        if src.kind is SourceKind.RULE:
            raise ValueError(
                "rule sources are reserved for derive(); assert premises from a tool, document, human or assumption"
            )
        conf = self.trust.confidence_for(src) if confidence is None else _check_confidence(confidence)
        until = valid_until if valid_until is not None else (self.now() + ttl if ttl is not None else None)
        if until is not None and (until.tzinfo is None) != (self.now().tzinfo is None):
            kind = "naive" if self.now().tzinfo is None else "timezone-aware"
            raise ValueError(f"valid_until must be {kind}, like the belief base's clock")

        target = self._matching_node(key, value, allow_expired=True)
        created = target is None
        if target is None:
            target = self._new_node(key, value, source=src, claim=claim, confidence=conf, metadata=metadata)
        superseded = [n for n in self._in_nodes(key) if n is not target] if supersede else []
        for old in superseded:
            old.retracted, old.retract_reason = True, f"superseded by {target.ref}"
            self._hints[old.ref] = old.retract_reason
        renewing = not created and target.status is Status.OUT
        # The same source saying the same thing again (a re-read, a refresh) replaces its earlier
        # justification: it is not new evidence, and it must not pile up or count as confirmation.
        repeated = [] if created else [j for j in target.justifications if j.is_premise and j.source == src]
        positions = [target.justifications.index(j) for j in repeated]
        confirming = not created and not renewing and not repeated
        if created:
            self._hints[target.ref] = f"asserted by {src}"
        else:
            self._hints[target.ref] = f"renewed by {src}" if renewing or repeated else f"corroborated by {src}"
        for j in repeated:
            self._unlink(target, j)
        try:
            self._add_justification(
                target,
                kind=JustificationKind.PREMISE,
                source=src,
                confidence=conf,
                valid_until=until,
                half_life=half_life,
                extra_seeds=[n.ref for n in superseded],
                created=created,
            )
        except BaseException:
            for j, position in zip(repeated, positions, strict=True):
                self._link(target, j)
                target.justifications.remove(j)  # back where it was: support is chosen by position
                target.justifications.insert(position, j)
            for old in superseded:
                old.retracted, old.retract_reason = False, ""
            self._relabel([target.ref, *(n.ref for n in superseded)] if repeated else [n.ref for n in superseded])
            raise
        if confirming:  # only once the assertion has stuck
            self._credit_confirmation(target, src)
        for old in superseded:
            self._log("retract", old.ref, old.retract_reason)
        if created:
            self._log("assert", target.ref, f"= {format_value(value)} by {src}")
        else:
            self._log("renew" if renewing or repeated else "support", target.ref, f"by {src}")
        return target.belief

    def assume(self, key: str, value: Any, *, by: str = "user", **kwargs: Any) -> Belief:
        """Assert a working hypothesis. Assumptions are premises the verifier flags as ungrounded."""
        return self.assert_(key, value, source=Source.assumption(by), **kwargs)

    def add_document(self, name: str, text: str) -> None:
        """Register a document so beliefs can cite it and the verifier can check those citations.

        Registering a name again replaces its text. Beliefs cited from the old text are not
        retracted; the verifier re-checks their quotes against the new text.
        """
        if not name:
            raise ValueError("document name must not be empty")
        self._documents[name] = text

    @property
    def documents(self) -> Mapping[str, str]:
        return dict(self._documents)

    def cite(
        self,
        key: str,
        value: Any,
        *,
        document: str,
        quote: str,
        claim: str = "",
        confidence: float | None = None,
        check_value: bool = True,
        **kwargs: Any,
    ) -> Belief:
        """Assert a premise grounded in a span of a registered document.

        The quote must appear in the document (after whitespace and case normalization). With
        ``check_value`` the value itself must also be stated in the quote, so a model cannot cite
        a real sentence while extracting a number that isn't in it.
        """
        if document not in self._documents:
            raise CitationError(f"unknown document {document!r}; register it with add_document()")
        if not contains_quote(self._documents[document], quote):
            raise CitationError(f"quote not found in {document!r}: {quote!r}")
        if check_value and isinstance(value, (int, float, str)) and not value_in_text(value, quote):
            raise CitationError(f"value {format_value(value)} is not stated in the quoted text: {quote!r}")
        return self.assert_(
            key, value, source=Source.document(document, quote=quote), claim=claim, confidence=confidence, **kwargs
        )

    # ======================================================================================
    # Deriving conclusions
    # ======================================================================================

    def register_rule(self, r: Rule, *, replace: bool = False) -> Rule:
        """Make a rule available for derivation, automatic re-derivation and proof replay.

        A different rule with the same name is an error unless ``replace=True`` (for example when
        re-running a notebook cell). Beliefs already derived keep their values until an input
        changes; derive them again to apply the new function now.
        """
        existing = self._rules.get(r.name)
        if existing is not None and existing.fn is not r.fn and not replace:
            raise RuleError(f"a different rule named {r.name!r} is already registered; pass replace=True to replace it")
        self._rules[r.name] = r
        return r

    @property
    def rules(self) -> Mapping[str, Rule]:
        return dict(self._rules)

    def derive(
        self,
        key: str,
        rule: Rule | Callable[..., Any] | str,
        *inputs: str,
        unless: Iterable[str] = (),
        claim: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> Belief:
        """Derive ``key`` by applying a deterministic rule to the current values of ``inputs``.

        ``inputs`` are keys (current believed value) or refs (a specific revision). ``unless``
        makes the derivation non-monotonic: it holds only while none of those keys is believed.
        When an input later changes, :meth:`propagate` re-runs the rule automatically.
        """
        self.refresh()  # never derive from evidence that has expired since the last check
        r = self._coerce_rule(rule)
        nodes = [self._resolve_antecedent(item) for item in inputs]
        try:
            value = r(*(n.belief.value for n in nodes))
        except Exception as exc:
            raise RuleError(f"rule {r.name!r} failed for {key!r}: {exc}") from exc
        return self._conclude(
            key,
            value,
            kind=JustificationKind.RULE,
            antecedents=tuple(n.ref for n in nodes),
            inputs=tuple(n.belief.key for n in nodes),
            unless=_keys(unless),
            source=Source.rule(r.name),
            rule=r.name,
            confidence=r.confidence,
            claim=claim,
            metadata=metadata,
            hint=f"derived via rule:{r.name}",
        )

    def justify(
        self,
        key: str,
        value: Any,
        *,
        antecedents: Iterable[str],
        source: Source | str,
        claim: str = "",
        formula: str | None = None,
        confidence: float | None = None,
        unless: Iterable[str] = (),
        inputs: Iterable[str] | None = None,
        note: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> Belief:
        """Record a conclusion justified by ``antecedents``, typically proposed by a model.

        Every antecedent must currently be believed. When ``formula`` is given it is evaluated
        against the antecedent values and must reproduce ``value``, otherwise
        :class:`FormulaError` is raised and nothing is recorded.

        ``confidence`` is the certainty of this particular step (a stated confidence, or a
        self-consistency score) between 0 and 1, defaulting to 1. The source's own reliability is
        applied on top when confidence is computed; see :meth:`confidence`.
        """
        src = Source.parse(source)
        self.refresh()  # never conclude from evidence that has expired since the last check
        nodes = [self._resolve_antecedent(item) for item in _keys(antecedents)]
        if formula is not None:
            values = {n.belief.key: n.belief.value for n in nodes}
            missing = [k for k in formula_keys(formula) if k not in values]
            if missing:
                raise FormulaError(f"formula references {', '.join(missing)}, which are not antecedents")
            computed = evaluate(formula, values)
            if not _numeric_match(computed, value):
                raise FormulaError(
                    f"formula {formula!r} evaluates to {format_value(computed)}, not {format_value(value)}"
                )
        conf = 1.0 if confidence is None else _check_confidence(confidence)
        return self._conclude(
            key,
            value,
            kind=JustificationKind.MODEL,
            antecedents=tuple(n.ref for n in nodes),
            inputs=_keys(inputs) if inputs is not None else tuple(n.belief.key for n in nodes),
            unless=_keys(unless),
            source=src,
            formula=formula,
            confidence=conf,
            claim=claim,
            note=note,
            metadata=metadata,
            hint=f"derived by {src}",
        )

    # ======================================================================================
    # Retraction
    # ======================================================================================

    def retract(self, key_or_ref: str, *, reason: str = "", fault: str = "none") -> list[Belief]:
        """Withdraw a belief. Everything that depended on it goes ``OUT`` immediately.

        A key retracts every ``IN`` revision of that key; a ref (``key@2``) retracts exactly that
        revision. Call :meth:`propagate` afterwards to re-derive what can be re-derived and to get
        the diff of what changed.

        ``fault`` says whose mistake it was, which is what the trust ledger learns from:

        * ``"none"`` (default): nobody's. The world changed, e.g. figures were restated.
        * ``"source"``: the value was wrong when it was given. Every source accountable for the
          belief (its premise sources, or the model behind an unverified claim) is recorded as
          wrong, and its reliability drops.
        """
        if fault not in ("none", "source"):
            raise ValueError(f"fault must be 'none' or 'source', got {fault!r}")
        key, revision = parse_ref(key_or_ref)
        if revision is not None:
            node = self._node(key_or_ref)
            if node.retracted:
                return []  # Already withdrawn; retracting again must not blame its sources twice.
            nodes = [node]
        else:
            self._require_key(key)
            nodes = self._in_nodes(key)
            if not nodes:
                raise NotBelievedError(f"{key!r} is not currently believed, so there is nothing to retract")
        if fault == "source":
            if not any(self._accountable_sources(n) for n in nodes):
                raise ValueError(
                    f"{key_or_ref!r} is derived by a rule or a verified formula, so no source is at fault; "
                    "retract the input that was wrong instead"
                )
            self._record_outcomes(nodes, correct=False, reason=reason or "retracted as wrong")
        return self._retract_nodes(nodes, reason)

    def restore(self, key_or_ref: str) -> Belief:
        """Undo a retraction (the latest retracted revision when given a key)."""
        key, revision = parse_ref(key_or_ref)
        if revision is not None:
            node = self._node(key_or_ref)
        else:
            candidates = [self._nodes[r] for r in self._refs(key) if self._nodes[r].retracted]
            if not candidates:
                raise NotBelievedError(f"{key!r} has no retracted revision to restore")
            node = candidates[-1]
        node.retracted, node.retract_reason = False, ""
        self._hints[node.ref] = "restored"
        self._relabel([node.ref])
        self._log("restore", node.ref)
        return node.belief

    def _retract_nodes(self, nodes: list[_Node], reason: str) -> list[Belief]:
        for node in nodes:
            node.retracted, node.retract_reason = True, reason
            self._hints[node.ref] = f"retracted: {reason}" if reason else "retracted"
            self._log("retract", node.ref, reason)
        self._relabel([n.ref for n in nodes])
        return [n.belief for n in nodes]

    # ======================================================================================
    # Propagation and re-derivation
    # ======================================================================================

    def propagate(
        self,
        *,
        rederive: Rederiver | None = None,
        include_kept: bool = False,
        max_rounds: int = 100,
    ) -> Propagation:
        """Re-derive what lost support, then report everything that changed since the last call.

        Rule-derived beliefs are recomputed automatically. Model-derived beliefs are recomputed
        through ``rederive`` (an :class:`~corollary.Agent` passes its own); without it they are
        reported as :attr:`Propagation.pending`.

        If a re-derived value equals the old one, the old revision simply gains a new
        justification and comes back ``IN``, which stops the cascade there (early cutoff).
        """
        self.refresh()
        rederived: list[Belief] = []
        waiting: dict[str, str] = {}
        for _ in range(max_rounds):
            progressed = False
            candidates = self._rederive_candidates()
            waiting_keys = {n.belief.key for n in candidates}
            for node in candidates:
                if node.status is Status.IN or self._in_nodes(node.belief.key):
                    waiting_keys.discard(node.belief.key)
                    continue
                result, why = self._try_rederive(node, rederive, waiting_keys)
                if result is not None:
                    waiting_keys.discard(node.belief.key)
                if result is not None:
                    rederived.append(result)
                    waiting.pop(node.ref, None)
                    progressed = True
                else:
                    waiting[node.ref] = why
            if not progressed:
                break
        pending = [
            Pending(self._nodes[ref].belief, why)
            for ref, why in waiting.items()
            if self._nodes[ref].status is Status.OUT and not self._in_nodes(self._nodes[ref].belief.key)
        ]
        changes = self._drain(include_kept=include_kept)
        return Propagation(changes=changes, pending=pending, conflicts=self.conflicts(), rederived=rederived)

    def changes(self, *, include_kept: bool = False) -> list[Change]:
        """Net status changes since the last call to :meth:`changes` or :meth:`propagate`.

        Clears the buffer. Unlike :meth:`propagate`, nothing is re-derived.
        """
        return self._drain(include_kept=include_kept)

    def refresh(self) -> list[Belief]:
        """Re-check validity windows against the clock. Returns beliefs that just expired."""
        now = self.now()
        if self._next_expiry is None or now < self._next_expiry:
            return []  # nothing can have expired yet: the common case, and it costs nothing
        # Forget beliefs that no longer have any evidence with a validity window.
        self._expiring = {
            ref for ref in self._expiring if any(j.valid_until is not None for j in self._nodes[ref].justifications)
        }
        self._next_expiry = min(
            (
                j.valid_until
                for ref in self._expiring
                for j in self._nodes[ref].justifications
                if j.valid_until is not None and j.valid_until > now
            ),
            default=None,
        )
        seeds = [
            ref
            for ref in self._expiring
            if self._nodes[ref].status is Status.IN
            and self._nodes[ref].support is not None
            and self._nodes[ref].support.expired(now)  # type: ignore[union-attr]
        ]
        if not seeds:
            return []
        for ref in seeds:
            self._hints.setdefault(ref, "expired")
        self._relabel(seeds)
        return [self._nodes[r].belief for r in seeds if self._nodes[r].status is Status.OUT]

    def stale(self) -> list[Belief]:
        """Latest revisions that are ``OUT`` only because all of their evidence expired."""
        now = self.now()
        result = []
        for key, refs in self._revisions.items():
            node = self._nodes[refs[-1]]
            if node.status is Status.IN or node.retracted or self._in_nodes(key):
                continue
            premises = [j for j in node.justifications if j.is_premise]
            if premises and all(j.expired(now) for j in premises) and not node.derived:
                result.append(node.belief)
        return result

    def _rederive_candidates(self) -> list[_Node]:
        nodes = []
        for refs in self._revisions.values():
            node = self._nodes[refs[-1]]
            if node.status is Status.OUT and not node.retracted and any(j.rederivable for j in node.justifications):
                nodes.append(node)
        # Dependency order: a belief comes after every candidate among its recipe inputs. Timestamps
        # only break ties; they can't be relied on for order (coarse clocks give equal values).
        by_key = {n.belief.key: n for n in nodes}
        deps = {
            key: {k for j in n.justifications if j.rederivable for k in j.inputs if k in by_key and k != key}
            for key, n in by_key.items()
        }

        def tie_break(key: str) -> tuple[datetime, str]:
            return by_key[key].belief.created_at, key

        # Kahn's algorithm, always taking the earliest ready candidate: linear in the graph size.
        users: dict[str, list[str]] = defaultdict(list)
        for key, needed in deps.items():
            for dep in needed:
                users[dep].append(key)
        waiting = {key: len(needed) for key, needed in deps.items()}
        ready = [(tie_break(k), k) for k, n in waiting.items() if n == 0]
        heapq.heapify(ready)
        ordered: list[_Node] = []
        placed: set[str] = set()
        while len(placed) < len(by_key):
            if not ready:  # a cycle among candidates: release its earliest member
                key = min((k for k in by_key if k not in placed), key=tie_break)
                heapq.heappush(ready, (tie_break(key), key))
            _, key = heapq.heappop(ready)
            if key in placed:
                continue
            placed.add(key)
            ordered.append(by_key[key])
            for user in users[key]:
                waiting[user] -= 1
                if waiting[user] == 0 and user not in placed:
                    heapq.heappush(ready, (tie_break(user), user))
        return ordered

    def _try_rederive(
        self, node: _Node, rederive: Rederiver | None, waiting_keys: AbstractSet[str] = frozenset()
    ) -> tuple[Belief | None, str]:
        why = "no re-derivable justification"
        for just in reversed(node.justifications):
            if not just.rederivable:
                continue
            defeaters = [k for k in just.unless if self._key_in(k)]
            if defeaters:
                return None, f"defeated by: {', '.join(defeaters)}"
            inputs: dict[str, _Node] = {}
            missing: list[str] = []
            for key in just.inputs:
                try:
                    inputs[key] = self._current_node(key)
                except (UnknownBeliefError, NotBelievedError, UnresolvedConflictError):
                    missing.append(key)
            derived: Derived | None
            if just.kind is JustificationKind.RULE:
                if missing:
                    why = f"waiting for: {', '.join(missing)}"
                    continue
                r = self._rules.get(just.rule or "")
                if r is None:
                    why = f"rule {just.rule!r} is not registered"
                    continue
                try:
                    value = r(*(inputs[k].belief.value for k in just.inputs))
                except Exception as exc:
                    why = f"rule {r.name!r} failed: {exc}"
                    continue
                derived = Derived(value=value)
            else:
                # An input that is itself about to be re-derived must come first; otherwise the model
                # would re-derive from an incomplete context.
                blocked = [k for k in missing if k in waiting_keys]
                if blocked:
                    why = f"waiting for: {', '.join(blocked)}"
                    continue
                if rederive is None:
                    why = "needs model re-derivation" + (f" (missing: {', '.join(missing)})" if missing else "")
                    continue
                request = Rederivation(
                    belief=node.belief,
                    justification=just,
                    inputs={k: n.belief for k, n in inputs.items()},
                    missing=tuple(missing),
                )
                derived = rederive(request)
                if derived is None:
                    why = "model could not re-derive it"
                    continue
                if derived.antecedents is not None and not set(derived.antecedents) <= set(inputs):
                    why = "re-deriver cited beliefs outside its inputs"
                    continue
            return self._apply_rederivation(node, just, derived, inputs), ""
        return None, why

    def _apply_rederivation(
        self, node: _Node, just: Justification, derived: Derived, inputs: Mapping[str, _Node]
    ) -> Belief:
        keys = derived.antecedents if derived.antecedents is not None else tuple(k for k in just.inputs if k in inputs)
        antecedents = tuple(inputs[k].ref for k in keys)
        if values_equal(derived.value, node.belief.value):
            target, hint = node, "re-derived, unchanged"
        else:
            target = self._new_node(
                node.belief.key,
                derived.value,
                source=node.belief.source,
                claim=derived.claim or node.belief.claim,
                confidence=node.belief.confidence,
                metadata=node.belief.metadata,
            )
            hint = f"re-derived: {format_value(derived.value)}"
        self._hints[target.ref] = hint
        self._add_justification(
            target,
            kind=just.kind,
            antecedents=antecedents,
            unless=just.unless,
            inputs=just.inputs,
            source=just.source,
            rule=just.rule,
            formula=derived.formula if derived.formula is not None else just.formula,
            confidence=derived.confidence if derived.confidence is not None else just.confidence,
            note=just.note,
            created=target is not node,
        )
        self._log("rederive", target.ref, f"= {format_value(derived.value)}")
        return target.belief

    def _drain(self, *, include_kept: bool) -> list[Change]:
        changes: list[Change] = []
        for ref, before in self._baseline.items():
            node = self._nodes.get(ref)
            if node is None:
                continue
            after = node.status
            if before is after or (before is None and after is Status.OUT):
                continue
            kind = ChangeKind.IN if after is Status.IN else ChangeKind.OUT
            changes.append(Change(kind, node.belief, self._reasons.get(ref, "")))
        if include_kept:
            causes = list(dict.fromkeys(c.key for c in changes if (n := self._nodes[c.ref]).retracted or not n.derived))
            reason = f"independent of {', '.join(causes)}" if causes else "unaffected"
            for refs in self._revisions.values():
                for ref in refs:
                    node = self._nodes[ref]
                    if ref not in self._baseline and node.status is Status.IN and node.derived:
                        changes.append(Change(ChangeKind.KEPT, node.belief, reason))
        self._baseline.clear()
        return changes

    # ======================================================================================
    # Conflicts
    # ======================================================================================

    def add_constraint(
        self,
        constraint: Constraint | str,
        keys: Iterable[str] | None = None,
        predicate: Callable[..., bool] | None = None,
        *,
        description: str = "",
    ) -> Constraint:
        """Register an invariant. ``add_constraint("margin", ["margin"], lambda m: m <= 100)``."""
        if isinstance(constraint, str):
            if keys is None or predicate is None:
                raise ValueError("add_constraint(name, keys, predicate) requires keys and a predicate")
            constraint = Constraint(constraint, _keys(keys), predicate, description)
        self._constraints[constraint.name] = constraint
        return constraint

    def conflicts(self) -> list[Conflict]:
        """All open conflicts: keys with incompatible ``IN`` values, and violated constraints."""
        found: list[Conflict] = []
        for key, nodes in self._value_conflicts():
            beliefs = tuple(n.belief for n in nodes)
            found.append(
                Conflict(
                    id=f"value:{key}",
                    kind=ConflictKind.VALUE,
                    subject=key,
                    beliefs=beliefs,
                    description=f"{len(beliefs)} incompatible values ({describe_values(beliefs)})",
                    proofs=tuple(Proof.build(self, b.ref) for b in beliefs),
                )
            )
        for constraint in self._constraints.values():
            try:
                nodes = [self._current_node(k) for k in constraint.keys]
            except (UnknownBeliefError, NotBelievedError, UnresolvedConflictError):
                continue
            values = [n.belief.value for n in nodes]
            try:
                ok = bool(constraint.predicate(*values))
                detail = constraint.description or f"constraint {constraint.name!r} violated"
            except Exception as exc:
                ok, detail = False, f"constraint {constraint.name!r} raised {exc!r}"
            if ok:
                continue
            beliefs = tuple(n.belief for n in nodes)
            found.append(
                Conflict(
                    id=f"constraint:{constraint.name}",
                    kind=ConflictKind.CONSTRAINT,
                    subject=constraint.name,
                    beliefs=beliefs,
                    description=detail,
                    proofs=tuple(Proof.build(self, b.ref) for b in beliefs),
                )
            )
        return found

    def conflicted_keys(self) -> set[str]:
        """Keys involved in a *value* conflict (constraint conflicts don't block usage)."""
        return {key for key, _ in self._value_conflicts()}

    def _value_conflicts(self) -> Iterator[tuple[str, list[_Node]]]:
        """Keys with incompatible ``IN`` values, and those revisions. Only keys with several
        revisions can conflict, so the common single-revision key costs one length check."""
        for key, refs in self._revisions.items():
            if len(refs) < 2:
                continue
            nodes = self._in_nodes(key)
            if len(nodes) > 1 and any(not values_equal(nodes[0].belief.value, n.belief.value) for n in nodes[1:]):
                yield key, nodes

    def resolve(
        self,
        conflict: Conflict,
        *,
        keep: str | Belief | Iterable[str | Belief] | None = None,
        retract: str | Belief | Iterable[str | Belief] | None = None,
        reason: str = "",
        learn: bool = False,
    ) -> Resolution:
        """Resolve a conflict by keeping some sides (retracting the rest) or retracting given sides.

        With ``learn=True`` the decision is treated as ground truth (a person checked it): the
        sources of retracted sides are recorded as wrong in the trust ledger, and for value
        conflicts the sources of the kept side as right. Leave it off for policy decisions, or a
        policy would end up reinforcing itself.
        """
        if (keep is None) == (retract is None):
            raise ValueError("pass exactly one of keep= or retract=")
        chosen: set[str] = set()
        for item in _as_list(keep if keep is not None else retract):
            sides = [r for r in conflict.refs if isinstance(item, str) and parse_ref(r)[0] == item]
            if len(sides) > 1:
                raise ValueError(
                    f"{item!r} is on several sides of conflict {conflict.id} ({', '.join(sides)}); pass a ref"
                )
            chosen.add(sides[0] if sides else self._node(_as_ref(self, item)).ref)
        unknown = chosen - set(conflict.refs)
        if unknown:
            raise ValueError(f"{', '.join(sorted(unknown))} are not part of conflict {conflict.id}")
        refs = [r for r in conflict.refs if (r not in chosen) == (keep is not None)]
        why = reason or f"resolved conflict {conflict.id}"
        resolution = Resolution(conflict.id, tuple(refs), why, authoritative=learn)
        self._apply_resolution(conflict, resolution)
        return resolution

    def _apply_resolution(self, conflict: Conflict, resolution: Resolution) -> list[str]:
        live = [r for r in resolution.retract if self._nodes[r].status is Status.IN]
        if not live:
            return []
        if resolution.authoritative:
            self._record_outcomes([self._nodes[r] for r in live], correct=False, reason=resolution.reason)
            if conflict.kind is ConflictKind.VALUE:
                kept = [self._nodes[r] for r in conflict.refs if r not in resolution.retract]
                self._record_outcomes(kept, correct=True, reason=resolution.reason)
        self._retract_nodes([self._nodes[r] for r in live], resolution.reason)
        self._log("resolve", conflict.id, f"retracted {', '.join(live)}")
        return live

    def resolve_conflicts(self, resolver: Resolver, *, max_rounds: int = 10) -> list[Resolution]:
        """Apply ``resolver`` to every open conflict until none can be resolved."""
        applied: list[Resolution] = []
        for _ in range(max_rounds):
            progressed = False
            for conflict in self.conflicts():
                decision = resolver(conflict, self)
                if decision is None or not decision.retract:
                    continue
                unknown = sorted(set(decision.retract) - set(conflict.refs))
                if unknown:
                    raise ValueError(
                        f"resolver {resolver!r} retracts {', '.join(unknown)}, "
                        f"which are not part of conflict {conflict.id}"
                    )
                if not self._apply_resolution(conflict, decision):
                    continue
                applied.append(decision)
                progressed = True
            if not progressed:
                break
        return applied

    # ======================================================================================
    # Queries
    # ======================================================================================

    def status(self, key_or_ref: str) -> Status:
        """``IN`` if the ref is believed, or if any revision of the key is believed."""
        key, revision = parse_ref(key_or_ref)
        if revision is not None:
            return self._node(key_or_ref).status
        self._require_key(key)
        return Status.IN if self._key_in(key) else Status.OUT

    def get(self, key: str, default: Any = None) -> Belief | Any:
        """The currently believed revision of ``key``, or ``default`` if none.

        Raises :class:`UnresolvedConflictError` if the key has incompatible ``IN`` values.
        """
        try:
            return self._current_node(key).belief
        except (UnknownBeliefError, NotBelievedError):
            return default

    def __getitem__(self, key: str) -> Belief:
        return self._current_node(key).belief

    def value(self, key: str) -> Any:
        """Shortcut for ``kb[key].value``."""
        return self._current_node(key).belief.value

    def __contains__(self, key_or_ref: object) -> bool:
        if not isinstance(key_or_ref, str):
            return False
        key, revision = parse_ref(key_or_ref)
        if revision is not None:
            return key_or_ref in self._nodes and self._nodes[key_or_ref].status is Status.IN
        return self._key_in(key)

    def latest(self, key: str) -> Belief:
        """The most recent revision of ``key`` regardless of status."""
        return self._nodes[self._refs(key)[-1]].belief

    def revisions(self, key: str) -> list[Belief]:
        return [self._nodes[r].belief for r in self._refs(key)]

    def keys(self, status: Status | None = Status.IN) -> list[str]:
        """Keys with a revision of the given status (``None`` for every key)."""
        if status is None:
            return list(self._revisions)
        if status is Status.IN:
            return [k for k in self._revisions if self._key_in(k)]
        return [k for k in self._revisions if not self._key_in(k)]

    def beliefs(self, status: Status | None = Status.IN) -> list[Belief]:
        """Belief revisions with the given status (``None`` for all), in creation order."""
        return [
            n.belief
            for refs in self._revisions.values()
            for n in (self._nodes[r] for r in refs)
            if status is None or n.status is status
        ]

    def __iter__(self) -> Iterator[Belief]:
        return iter(self.beliefs(Status.IN))

    def __len__(self) -> int:
        return sum(1 for n in self._nodes.values() if n.status is Status.IN)

    def confidence(self, key_or_ref: str) -> float:
        """Effective confidence of a belief, between 0 (``OUT``) and 1.

        * A premise counts ``reliability(source) * freshness``: the source's reliability is learned
          by the trust ledger, starting from the trust policy's prior, and evidence with a
          ``half_life`` fades with age.
        * Premises from independent origins combine by noisy-OR: ``1 - (1 - c1)(1 - c2)...``.
          Sources sharing an origin count once (the most confident one).
        * A derived step counts ``step * min(antecedents)``: rules and re-executed formulas are
          mechanical (rule-level trust); an unverified model step uses the model's learned
          reliability times its stated certainty.
        * With several valid justifications, the strongest one wins.

        See the confidence guide in the documentation for worked examples.
        """
        node = self._resolve(key_or_ref)
        now = self._confidence_time()
        epoch = (
            self._version,
            self.ledger.version,
            id(self.trust),
            self.trust.fingerprint(),
            now if self._time_dependent else None,
        )
        if epoch != self._conf_epoch:
            self._conf_cache, self._conf_epoch = {}, epoch
        if node.ref not in self._conf_cache:
            self._compute_confidence(node.ref, now)
        return self._conf_cache[node.ref]

    def reliability(self, source: Source | str) -> float:
        """A source's learned reliability, starting from the trust policy's prior for it."""
        src = Source.parse(source)
        return self.ledger.reliability(src, prior=self.trust.confidence_for(src), at=self.now())

    def record_outcome(self, source: Source | str, correct: bool, *, reason: str = "") -> None:
        """Tell the trust ledger that ``source`` turned out right or wrong (e.g. from a support
        ticket or an audit). Confidence of every belief from that source updates accordingly."""
        self.ledger.record(Source.parse(source), correct, at=self.now(), reason=reason)

    def faded(self, threshold: float | None = None) -> list[Belief]:
        """Believed premises whose decaying evidence has faded below ``threshold``.

        Defaults to the trust policy's ``min_confidence``, the level below which the projector
        hides beliefs from the model. :meth:`Agent.reverify` refreshes these.
        """
        limit = self.trust.min_confidence if threshold is None else threshold
        result = []
        for refs in self._revisions.values():
            node = self._nodes[refs[-1]]
            if node.status is not Status.IN or node.derived:
                continue
            if any(j.half_life is not None for j in node.justifications) and self.confidence(node.ref) < limit:
                result.append(node.belief)
        return result

    def valid_until(self, key_or_ref: str) -> datetime | None:
        """Earliest expiry along the current support chain, or ``None`` if nothing expires."""
        node = self._resolve(key_or_ref)
        earliest: datetime | None = None
        stack, seen = [node], set()
        while stack:
            n = stack.pop()
            if n.ref in seen or n.support is None:
                continue
            seen.add(n.ref)
            if n.support.valid_until is not None and (earliest is None or n.support.valid_until < earliest):
                earliest = n.support.valid_until
            stack.extend(self._nodes[a] for a in n.support.antecedents)
        return earliest

    def support(self, key_or_ref: str) -> Justification | None:
        """The justification currently making the belief ``IN`` (``None`` when ``OUT``)."""
        return self._resolve(key_or_ref).support

    def justifications(self, key_or_ref: str) -> list[Justification]:
        return list(self._resolve(key_or_ref).justifications)

    def dependents(self, key_or_ref: str, *, transitive: bool = True) -> list[Belief]:
        """Beliefs whose justifications use this one (directly, or anywhere downstream)."""
        start = self._resolve(key_or_ref)
        seen: dict[str, None] = {}
        frontier = deque([start.ref])
        while frontier:
            ref = frontier.popleft()
            for nxt in self._successors(ref):
                if nxt not in seen and nxt != start.ref:
                    seen[nxt] = None
                    if transitive:
                        frontier.append(nxt)
        return [self._nodes[r].belief for r in seen]

    def why_out(self, key_or_ref: str) -> str | None:
        """Why a belief is ``OUT`` (``None`` if it is ``IN``)."""
        node = self._resolve(key_or_ref)
        if node.status is Status.IN:
            return None
        return self._out_reason(node, node.justifications[-1] if node.justifications else None)

    def explain(self, key_or_ref: str) -> str:
        """A readable account of a belief: value, status, sources, support and dependents."""
        node = self._resolve(key_or_ref)
        b = node.belief
        lines = [f"{b.ref} = {format_value(b.value)}  [{node.status.value}, confidence {self.confidence(b.ref):.2f}]"]
        if b.claim:
            lines.append(f"  claim:      {b.claim}")
        if node.status is Status.OUT:
            lines.append(f"  why OUT:    {self.why_out(b.ref)}")
        for j in node.justifications:
            marker = "*" if j is node.support else " "
            parts = [j.describe()]
            if j.antecedents:
                parts.append("from " + ", ".join(j.antecedents))
            if j.unless:
                parts.append("unless " + ", ".join(j.unless))
            if j.formula:
                parts.append(f"formula {j.formula}")
            if j.valid_until:
                parts.append(f"valid until {j.valid_until.isoformat(timespec='seconds')}")
            if j.half_life:
                parts.append(f"half-life {j.half_life}, freshness {j.freshness(self.now()):.2f}")
            lines.append(f"  {marker} {j.id:<6} {j.kind.value:<8} " + "; ".join(parts))
        used_by = [d.ref for d in self.dependents(b.ref, transitive=False)]
        if used_by:
            lines.append(f"  used by:    {', '.join(used_by)}")
        return "\n".join(lines)

    def proof(self, *keys_or_refs: str) -> Proof:
        """Snapshot the support graph behind one or more beliefs."""
        if not keys_or_refs:
            raise ValueError("proof() needs at least one key or ref")
        return Proof.build(self, *keys_or_refs)

    @property
    def history(self) -> list[Event]:
        return list(self._history)

    def transcript(self) -> str:
        """The event history rendered as text: the message log, as a view of the belief base."""
        return "\n".join(str(e) for e in self._history)

    def __repr__(self) -> str:
        n_in = len(self)
        return f"<BeliefBase {len(self._revisions)} keys, {n_in} IN, {len(self._nodes) - n_in} OUT>"

    # ======================================================================================
    # Persistence
    # ======================================================================================

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible snapshot. Rules and constraints are referenced by name, not serialized.

        The trust ledger is included when this base owns it. A shared ledger (passed with
        ``ledger=``) belongs to all its bases, so save it separately with :meth:`TrustLedger.save`.
        """
        data = {
            "format": _FORMAT,
            "version": _FORMAT_VERSION,
            "beliefs": [
                {"belief": n.belief.to_dict(), "retracted": n.retracted, "retract_reason": n.retract_reason}
                for refs in self._revisions.values()
                for n in (self._nodes[r] for r in refs)
            ],
            "justifications": [j.to_dict() for j in self._justifications.values()],
            "documents": dict(self._documents),
            "reasons": dict(self._reasons),
            "history": [
                {"at": e.at.isoformat(), "action": e.action, "ref": e.ref, "detail": e.detail} for e in self._history
            ],
        }
        if self._owns_ledger:
            data["ledger"] = self.ledger.to_dict()
        return data

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        rules: Iterable[Rule] = (),
        constraints: Iterable[Constraint] = (),
        trust: TrustPolicy | None = None,
        clock: Callable[[], datetime] | None = None,
        ledger: TrustLedger | None = None,
    ) -> BeliefBase:
        """Rebuild a base from :meth:`to_dict`. Pass ``ledger=`` to attach a shared trust ledger;
        otherwise the snapshot's own ledger is restored, if it has one."""
        return cls._load_data(
            data, rules=rules, constraints=constraints, trust=trust, clock=clock, ledger=ledger, stacklevel=3
        )

    @classmethod
    def _load_data(
        cls,
        data: Mapping[str, Any],
        *,
        rules: Iterable[Rule] = (),
        constraints: Iterable[Constraint] = (),
        trust: TrustPolicy | None = None,
        clock: Callable[[], datetime] | None = None,
        ledger: TrustLedger | None = None,
        stacklevel: int,
    ) -> BeliefBase:
        if data.get("format") != _FORMAT:
            raise ValueError("not a Corollary belief base snapshot")
        version = data.get("version", 0)
        if isinstance(version, bool) or not isinstance(version, int):
            raise ValueError(f"snapshot version must be an integer, got {version!r}")
        if version < 0:
            raise ValueError(f"snapshot version {version} is invalid (must be >= 0)")
        if version > _FORMAT_VERSION:
            raise ValueError(f"snapshot version {version} is newer than this library supports")
        with _corrupt_snapshot_errors():
            saved_ledger = TrustLedger.from_dict(data["ledger"]) if ledger is None and data.get("ledger") else None
        # Built outside the guard: a mistake in the caller's own arguments is not a corrupt snapshot.
        kb = cls(trust=trust, clock=clock, rules=rules, constraints=constraints, ledger=saved_ledger or ledger)
        kb._owns_ledger = saved_ledger is not None or kb._owns_ledger
        with _corrupt_snapshot_errors():
            kb._restore(data, version)
        missing = sorted({j.rule for j in kb._justifications.values() if j.rule and j.rule not in kb._rules})
        if missing:
            warnings.warn(
                f"the snapshot uses rules that were not passed to load(): {', '.join(missing)}. Beliefs "
                "derived by them can't be re-derived or replayed by the verifier until you register them.",
                stacklevel=stacklevel,
            )
        return kb

    def _restore(self, data: Mapping[str, Any], version: int) -> None:
        for item in data["beliefs"]:
            belief = Belief.from_dict(item["belief"])
            node = _Node(belief, retracted=bool(item.get("retracted")), retract_reason=item.get("retract_reason", ""))
            self._nodes[belief.ref] = node
            self._revisions.setdefault(belief.key, []).append(belief.ref)
        max_id = 0
        for raw in data["justifications"]:
            j = Justification.from_dict(raw)
            if version < 2:
                j = self._migrate_v1_justification(j)
            self._link(self._nodes[j.conclusion], j)
            if j.id[1:].isdigit():
                max_id = max(max_id, int(j.id[1:]))
        self._ids = itertools.count(max_id + 1)
        self._documents = dict(data.get("documents", {}))
        self._history = [
            Event(datetime.fromisoformat(e["at"]), e["action"], e["ref"], e.get("detail", ""))
            for e in data.get("history", [])
        ]
        self._relabel(list(self._nodes))
        self._baseline.clear()
        self._hints.clear()
        self._reasons = dict(data.get("reasons", {}))  # relabeling on load is not a real change

    def _migrate_v1_justification(self, j: Justification) -> Justification:
        """Version 1 stored model steps with the model's trust folded into ``confidence``; version 2
        stores only the step's certainty and applies the model's (learned) reliability on top."""
        if j.kind is not JustificationKind.MODEL or j.source is None:
            return j
        base = self.trust.sources.get("rule", 1.0) if j.formula else self.trust.confidence_for(j.source)
        return dataclasses.replace(j, confidence=min(1.0, j.confidence / base) if base else j.confidence)

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write a JSON snapshot. Values must be JSON-serializable. The write is atomic: a crash
        mid-write leaves the previous snapshot intact."""
        write_atomically(path, json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | os.PathLike[str], **kwargs: Any) -> BeliefBase:
        """Load a snapshot written by :meth:`save`. Pass ``rules=`` to re-link derivation rules."""
        return cls._load_data(json.loads(Path(path).read_text(encoding="utf-8")), stacklevel=3, **kwargs)

    # ======================================================================================
    # Internals: graph construction
    # ======================================================================================

    def _new_node(
        self,
        key: str,
        value: Any,
        *,
        source: Source,
        claim: str,
        confidence: float,
        metadata: Mapping[str, Any] | None,
    ) -> _Node:
        refs = self._revisions.setdefault(key, [])
        belief = Belief(
            key=key,
            value=value,
            source=source,
            revision=len(refs) + 1,
            claim=claim,
            confidence=confidence,
            created_at=self.now(),
            metadata=dict(metadata or {}),
        )
        node = _Node(belief)
        self._nodes[belief.ref] = node
        refs.append(belief.ref)
        self._baseline.setdefault(belief.ref, None)
        return node

    def _discard_node(self, node: _Node) -> None:
        refs = self._revisions[node.belief.key]
        refs.remove(node.ref)
        if not refs:
            del self._revisions[node.belief.key]
        del self._nodes[node.ref]
        self._expiring.discard(node.ref)
        self._baseline.pop(node.ref, None)
        self._hints.pop(node.ref, None)

    def _link(self, node: _Node, j: Justification) -> None:
        self._version += 1
        if j.half_life is not None:
            self._decaying = True
        node.justifications.append(j)
        self._justifications[j.id] = j
        for a in j.antecedents:
            self._nodes[a].consumers.add(j.id)
        for k in j.unless:
            self._unless_consumers[k].add(j.id)
        if j.valid_until is not None:
            self._expiring.add(node.ref)
            if self._next_expiry is None or j.valid_until < self._next_expiry:
                self._next_expiry = j.valid_until

    def _unlink(self, node: _Node, j: Justification) -> None:
        self._version += 1
        node.justifications.remove(j)
        del self._justifications[j.id]
        for a in j.antecedents:
            self._nodes[a].consumers.discard(j.id)
        for k in j.unless:
            self._unless_consumers[k].discard(j.id)

    def _add_justification(
        self,
        node: _Node,
        *,
        kind: JustificationKind,
        antecedents: tuple[str, ...] = (),
        unless: tuple[str, ...] = (),
        inputs: tuple[str, ...] = (),
        source: Source | None = None,
        rule: str | None = None,
        formula: str | None = None,
        confidence: float = 1.0,
        valid_until: datetime | None = None,
        half_life: timedelta | None = None,
        note: str = "",
        extra_seeds: Iterable[str] = (),
        created: bool = False,
    ) -> Justification:
        for k in unless:
            validate_key(k)
        j = Justification(
            id=f"j{next(self._ids)}",
            conclusion=node.ref,
            kind=kind,
            antecedents=antecedents,
            unless=unless,
            inputs=inputs,
            source=source,
            rule=rule,
            formula=formula,
            confidence=confidence,
            valid_until=valid_until,
            half_life=half_life,
            note=note,
            created_at=self.now(),
        )
        self._link(node, j)
        seeds = [node.ref, *extra_seeds]
        try:
            self._relabel(seeds)
        except BaseException:
            # Leave the graph as it was, whatever went wrong: a half-linked node would break
            # every later query, and saving.
            self._unlink(node, j)
            if created:
                self._discard_node(node)
                seeds = [s for s in seeds if s in self._nodes]
            self._relabel(seeds)
            raise
        return j

    def _conclude(
        self,
        key: str,
        value: Any,
        *,
        kind: JustificationKind,
        antecedents: tuple[str, ...],
        inputs: tuple[str, ...],
        unless: tuple[str, ...],
        source: Source,
        confidence: float,
        claim: str,
        metadata: Mapping[str, Any] | None,
        hint: str,
        rule: str | None = None,
        formula: str | None = None,
        note: str = "",
    ) -> Belief:
        validate_key(key)
        target = self._matching_node(key, value, allow_expired=False)
        created = target is None
        if target is None:
            target = self._new_node(key, value, source=source, claim=claim, confidence=confidence, metadata=metadata)
            self._hints[target.ref] = hint
        self._add_justification(
            target,
            kind=kind,
            antecedents=antecedents,
            unless=unless,
            inputs=inputs,
            source=source,
            rule=rule,
            formula=formula,
            confidence=confidence,
            note=note,
            created=created,
        )
        self._log("derive" if created else "support", target.ref, f"= {format_value(value)} {hint}")
        return target.belief

    def _matching_node(self, key: str, value: Any, *, allow_expired: bool) -> _Node | None:
        """An existing revision with the same value that a new justification should attach to."""
        if key not in self._revisions:
            return None
        for node in reversed(self._in_nodes(key)):
            if values_equal(node.belief.value, value):
                return node
        if allow_expired and not self._in_nodes(key):
            latest = self._nodes[self._revisions[key][-1]]
            if not latest.retracted and values_equal(latest.belief.value, value):
                return latest
        return None

    def _coerce_rule(self, r: Rule | Callable[..., Any] | str) -> Rule:
        if isinstance(r, Rule):
            return self.register_rule(r)
        if isinstance(r, str):
            try:
                return self._rules[r]
            except KeyError:
                raise RuleError(f"unknown rule {r!r}; register it first") from None
        if callable(r):
            for existing in self._rules.values():
                if existing.fn is r:
                    return existing
            name = getattr(r, "__name__", "rule")
            taken = set(self._rules) | {j.rule for j in self._justifications.values() if j.rule}
            if name == "<lambda>" or name in taken:
                # Anonymous functions, and different functions sharing a name (closures made by one
                # factory), get a unique name: one no rule and no existing belief uses, so a loaded
                # snapshot's beliefs never get re-linked to the wrong function. Use named rules if the
                # base will be persisted, since only names survive save()/load().
                stem, n = ("lambda", 1) if name == "<lambda>" else (name, 2)  # the first one is plain `name`
                while f"{stem}:{n}" in taken:
                    n += 1
                name = f"{stem}:{n}"
            return self.register_rule(Rule(name=name, fn=r))
        raise TypeError(f"expected a Rule, a callable or a rule name, got {type(r).__name__}")

    def _log(self, action: str, ref: str, detail: str = "") -> None:
        self._history.append(Event(self.now(), action, ref, detail))

    # ======================================================================================
    # Internals: lookup
    # ======================================================================================

    def _refs(self, key: str) -> list[str]:
        try:
            return self._revisions[key]
        except KeyError:
            raise UnknownBeliefError(f"no belief named {key!r}") from None

    def _require_key(self, key: str) -> None:
        self._refs(key)

    def _node(self, ref: str) -> _Node:
        try:
            return self._nodes[ref]
        except KeyError:
            raise UnknownBeliefError(f"no belief revision {ref!r}") from None

    def _in_nodes(self, key: str) -> list[_Node]:
        return [self._nodes[r] for r in self._revisions.get(key, ()) if self._nodes[r].status is Status.IN]

    def _key_in(self, key: str) -> bool:
        return any(self._nodes[r].status is Status.IN for r in self._revisions.get(key, ()))

    def _current_node(self, key: str) -> _Node:
        """The single believed revision of ``key``. Raises if unknown, OUT or conflicted."""
        self._require_key(key)
        nodes = self._in_nodes(key)
        if not nodes:
            raise NotBelievedError(f"{key!r} is not currently believed ({self.why_out(self._revisions[key][-1])})")
        if any(not values_equal(nodes[0].belief.value, n.belief.value) for n in nodes[1:]):
            raise UnresolvedConflictError(
                f"{key!r} has {len(nodes)} incompatible believed values; resolve the conflict first"
            )
        return nodes[-1]

    def _resolve(self, key_or_ref: str) -> _Node:
        """Ref -> that revision; key -> the believed revision, else the latest one."""
        key, revision = parse_ref(key_or_ref)
        if revision is not None:
            return self._node(key_or_ref)
        refs = self._refs(key)
        nodes = self._in_nodes(key)
        return nodes[-1] if nodes else self._nodes[refs[-1]]

    def _resolve_antecedent(self, key_or_ref: str) -> _Node:
        key, revision = parse_ref(key_or_ref)
        if revision is None:
            return self._current_node(key)
        node = self._node(key_or_ref)
        if node.status is not Status.IN:
            raise NotBelievedError(f"{key_or_ref!r} is OUT and cannot support a conclusion")
        return node

    def _resolve_for_proof(self, key_or_ref: str) -> str:
        return self._resolve(key_or_ref).ref

    def _proof_justification(self, ref: str) -> Justification | None:
        node = self._nodes[ref]
        if node.support is not None:
            return node.support
        return node.justifications[-1] if node.justifications else None

    # ======================================================================================
    # Internals: labeling
    # ======================================================================================

    def _successors(self, ref: str) -> Iterator[str]:
        node = self._nodes[ref]
        for jid in node.consumers:
            yield self._justifications[jid].conclusion
        for jid in self._unless_consumers.get(node.belief.key, ()):
            yield self._justifications[jid].conclusion

    def _downstream(self, seeds: Iterable[str]) -> dict[str, None]:
        region: dict[str, None] = {}
        stack = list(seeds)
        while stack:
            ref = stack.pop()
            if ref in region or ref not in self._nodes:
                continue
            region[ref] = None
            stack.extend(self._successors(ref))
        return region

    def _valid(self, j: Justification, now: datetime) -> bool:
        if j.expired(now):
            return False
        if any(self._nodes[a].status is not Status.IN for a in j.antecedents):
            return False
        return not any(self._key_in(k) for k in j.unless)

    def _components(self, region: dict[str, None]) -> list[list[str]]:
        """Strongly connected components of the region, dependencies first (Tarjan, iterative)."""
        index: dict[str, int] = {}
        low: dict[str, int] = {}
        on_stack: set[str] = set()
        stack: list[str] = []
        components: list[list[str]] = []
        counter = 0
        for root in region:
            if root in index:
                continue
            index[root] = low[root] = counter
            counter += 1
            stack.append(root)
            on_stack.add(root)
            work: list[tuple[str, Iterator[str]]] = [(root, self._successors(root))]
            while work:
                v, successors = work[-1]
                advanced = False
                for w in successors:
                    if w not in region:
                        continue
                    if w not in index:
                        index[w] = low[w] = counter
                        counter += 1
                        stack.append(w)
                        on_stack.add(w)
                        work.append((w, self._successors(w)))
                        advanced = True
                        break
                    if w in on_stack:
                        low[v] = min(low[v], index[w])
                if advanced:
                    continue
                work.pop()
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[v])
                if low[v] == index[v]:
                    component = []
                    while True:
                        w = stack.pop()
                        on_stack.discard(w)
                        component.append(w)
                        if w == v:
                            break
                    components.append(component)
        components.reverse()
        return components

    def _relabel(self, seeds: Iterable[str]) -> None:
        self._version += 1
        region = self._downstream(seeds)
        if not region:
            return
        now = self.now()
        before = {ref: (self._nodes[ref].status, self._nodes[ref].support) for ref in region}
        try:
            order = self._label_components(region, now)
        except CircularDefeatError:
            for ref, (status, support) in before.items():
                self._nodes[ref].status, self._nodes[ref].support = status, support
            raise
        # Record changes in dependency order, so a diff reads from causes to consequences.
        for ref in order:
            old_status, old_support = before[ref]
            node = self._nodes[ref]
            if node.status is old_status:
                continue
            self._baseline.setdefault(ref, old_status)
            hint = self._hints.pop(ref, None)
            if node.status is Status.IN:
                self._reasons[ref] = hint or f"support restored via {node.support.describe() if node.support else '?'}"
            elif hint and hint.startswith(("retracted", "superseded")):
                self._reasons[ref] = hint
            else:
                self._reasons[ref] = self._out_reason(node, old_support)
        # Hints only describe the change that consumed them.
        for ref in region:
            self._hints.pop(ref, None)

    def _label_components(self, region: dict[str, None], now: datetime) -> list[str]:
        """Label every component in dependency order; return the refs in that order."""
        order: list[str] = []
        for component in self._components(region):
            order.extend(component)
            members = set(component)
            for ref in component:
                for j in self._nodes[ref].justifications:
                    for k in j.unless:
                        if any(r in members for r in self._revisions.get(k, ())):
                            raise CircularDefeatError(
                                f"{ref} would depend on the absence of {k!r}, which depends on {ref}"
                            )
            for ref in component:
                node = self._nodes[ref]
                node.status, node.support = Status.OUT, None
            changed = True
            while changed:
                changed = False
                for ref in component:
                    node = self._nodes[ref]
                    if node.status is Status.IN or node.retracted:
                        continue
                    for j in reversed(node.justifications):
                        if self._valid(j, now):
                            node.status, node.support = Status.IN, j
                            changed = True
                            break
        return order

    # ======================================================================================
    # Internals: confidence and the trust ledger
    # ======================================================================================

    @property
    def _time_dependent(self) -> bool:
        """Whether confidence changes with time alone: evidence decay, or a ledger that forgets."""
        return self._decaying or self.ledger.memory_half_life is not None

    def _confidence_time(self) -> datetime:
        # With decay, confidence depends on time. Millisecond granularity lets repeated queries
        # in one step share the cache without any meaningful loss of precision.
        now = self.now()
        return now.replace(microsecond=now.microsecond // 1000 * 1000) if self._time_dependent else now

    def _compute_confidence(self, root: str, now: datetime) -> None:
        """Fill the cache for ``root`` and everything its confidence depends on.

        Confidence is the least fixpoint of the per-belief equations, computed by iterating from
        zero. Every operation is monotone and never amplifies its inputs (min, max, products of
        values in [0, 1], noisy-OR over premises only), so support cycles cannot inflate values
        and iteration converges.
        """
        # Postorder: every belief comes after what it depends on, so an acyclic graph settles in
        # a single pass (plus one to confirm); only support cycles need more.
        order: list[str] = []
        seen: set[str] = set()
        stack: list[tuple[str, bool]] = [(root, False)]
        while stack:
            ref, expanded = stack.pop()
            if expanded:
                order.append(ref)
                continue
            if ref in seen or ref in self._conf_cache:
                continue
            seen.add(ref)
            stack.append((ref, True))
            node = self._nodes[ref]
            if node.status is Status.IN:
                for j in node.justifications:
                    if self._valid(j, now):
                        stack.extend((a, False) for a in j.antecedents if a not in seen)
        values = dict.fromkeys(order, 0.0)

        def get(ref: str) -> float:
            cached = self._conf_cache.get(ref)
            return cached if cached is not None else values.get(ref, 0.0)

        for _ in range(len(order) + 2):
            changed = False
            for ref in order:
                value = self._node_confidence(self._nodes[ref], get, now)
                if abs(value - values[ref]) > 1e-12:
                    values[ref] = value
                    changed = True
            if not changed:
                break
        self._conf_cache.update(values)

    def _node_confidence(self, node: _Node, get: Callable[[str], float], now: datetime) -> float:
        if node.status is not Status.IN:
            return 0.0
        by_origin: dict[str, float] = {}
        derived = 0.0
        for j in node.justifications:
            if not self._valid(j, now):
                continue
            if j.is_premise:
                origin = j.source.origin if j.source is not None else j.id
                by_origin[origin] = max(by_origin.get(origin, 0.0), self._premise_confidence(j, now))
            else:
                weakest = min((get(a) for a in j.antecedents), default=1.0)
                derived = max(derived, self._step_confidence(j, now) * weakest)
        if not by_origin:
            premises = 0.0
        elif self.trust.corroboration:
            doubt = 1.0
            for value in by_origin.values():
                doubt *= 1.0 - value
            premises = 1.0 - doubt
        else:
            premises = max(by_origin.values())
        return max(premises, derived)

    def _premise_confidence(self, j: Justification, now: datetime) -> float:
        base = j.confidence
        if j.source is not None:
            base = self.ledger.reliability(j.source, prior=base, at=now)
        return base * j.freshness(now)

    def _step_confidence(self, j: Justification, now: datetime) -> float:
        if j.kind is JustificationKind.RULE:
            return j.confidence
        if j.formula is not None or j.source is None:
            # A formula the runtime re-executed is a mechanical step.
            return self.trust.sources.get("rule", 1.0) * j.confidence
        prior = self.trust.confidence_for(j.source)
        return self.ledger.reliability(j.source, prior=prior, at=now) * j.confidence

    def _accountable_sources(self, node: _Node) -> list[Source]:
        """Sources whose mistake a wrong belief would be: premise sources, and the model behind
        an unverified claim. Rules and re-executed formulas are never at fault."""
        found: dict[str, Source] = {}
        for j in node.justifications:
            if j.source is None:
                continue
            if j.is_premise or (j.kind is JustificationKind.MODEL and j.formula is None):
                found.setdefault(j.source.id, j.source)
        return list(found.values())

    def _record_outcomes(self, nodes: Iterable[_Node], *, correct: bool, reason: str) -> None:
        now = self.now()
        for node in nodes:
            for source in self._accountable_sources(node):
                self.ledger.record(source, correct, at=now, reason=f"{node.ref}: {reason}")

    def _credit_confirmation(self, node: _Node, source: Source) -> None:
        """An independent source agreeing with an existing premise is evidence both were right."""
        now = self.now()
        others = {
            j.source.id: j.source
            for j in node.justifications
            if j.is_premise and j.source is not None and j.source.origin != source.origin and self._valid(j, now)
        }
        if not others:
            return
        reason = f"{node.ref}: independently confirmed"
        for other in others.values():
            self.ledger.record(other, True, at=now, reason=reason)
        self.ledger.record(source, True, at=now, reason=reason)

    def _out_reason(self, node: _Node, support: Justification | None) -> str:
        if node.retracted:
            return f"retracted: {node.retract_reason}" if node.retract_reason else "retracted"
        if support is None:
            return "no valid justification"
        if support.expired(self.now()):
            return "expired"
        lost = list(
            dict.fromkeys(
                self._nodes[a].belief.key for a in support.antecedents if self._nodes[a].status is not Status.IN
            )
        )
        defeaters = [k for k in support.unless if self._key_in(k)]
        parts = []
        if lost:
            parts.append(f"lost support: {', '.join(lost)}")
        if defeaters:
            parts.append(f"defeated by: {', '.join(defeaters)}")
        return "; ".join(parts) or "no valid justification"


# ==========================================================================================
# Helpers
# ==========================================================================================


def _check_confidence(value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"confidence must be between 0 and 1, got {value}")
    return float(value)


def _numeric_match(computed: float, stated: Any) -> bool:
    if isinstance(stated, bool) or not isinstance(stated, (int, float)):
        return False
    return values_equal(computed, stated, rel_tol=1e-6, abs_tol=1e-9)


@contextlib.contextmanager
def _corrupt_snapshot_errors() -> Iterator[None]:
    """Report a snapshot whose data has the wrong shape as one clear error."""
    try:
        yield
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ValueError(f"corrupt Corollary snapshot: missing or malformed {exc}") from exc


def _keys(items: str | Iterable[str]) -> tuple[str, ...]:
    """Keys from an iterable, treating a lone string as one key rather than as its characters."""
    return (items,) if isinstance(items, str) else tuple(items)


def _as_list(items: str | Belief | Iterable[str | Belief] | None) -> list[str | Belief]:
    if items is None:
        return []
    if isinstance(items, (str, Belief)):
        return [items]
    return list(items)


def _as_ref(kb: BeliefBase, item: str | Belief) -> str:
    if isinstance(item, Belief):
        return item.ref
    return kb._resolve(item).ref
