"""Proofs: the justification subgraph behind one or more beliefs."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from .belief import Belief, Status, format_value, utcnow
from .justification import Justification

if TYPE_CHECKING:
    from .kernel import BeliefBase
    from .verify import Check, VerificationReport


@dataclass(frozen=True)
class ProofStep:
    """One belief in a proof, with the justification that supports it."""

    belief: Belief
    status: Status
    confidence: float
    justification: Justification | None

    @property
    def ref(self) -> str:
        return self.belief.ref

    @property
    def antecedents(self) -> tuple[str, ...]:
        return self.justification.antecedents if self.justification else ()

    @property
    def is_premise(self) -> bool:
        return self.justification is None or self.justification.is_premise

    def to_dict(self) -> dict[str, Any]:
        return {
            "belief": self.belief.to_dict(),
            "status": self.status.value,
            "confidence": self.confidence,
            "justification": self.justification.to_dict() if self.justification else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProofStep:
        just = data.get("justification")
        return cls(
            belief=Belief.from_dict(data["belief"]),
            status=Status(data["status"]),
            confidence=float(data["confidence"]),
            justification=Justification.from_dict(just) if just else None,
        )


@dataclass(frozen=True)
class ProofDiff:
    """Difference between two proofs, keyed by belief key."""

    added: tuple[str, ...]
    removed: tuple[str, ...]
    changed: tuple[tuple[str, Any, Any], ...]
    """``(key, old value, new value)`` for beliefs whose value changed."""
    status_changed: tuple[tuple[str, Status, Status], ...] = ()
    """``(key, old status, new status)``, e.g. a step that went ``OUT`` after a retraction."""

    @property
    def empty(self) -> bool:
        return not (self.added or self.removed or self.changed or self.status_changed)

    def __str__(self) -> str:
        lines = [f"+ {k}" for k in self.added]
        lines += [f"- {k}" for k in self.removed]
        lines += [f"~ {k}: {format_value(a)} -> {format_value(b)}" for k, a, b in self.changed]
        lines += [f"! {k}: {a.value} -> {b.value}" for k, a, b in self.status_changed]
        return "\n".join(lines) if lines else "(identical)"


@dataclass(frozen=True)
class Proof:
    """The graph of beliefs a conclusion follows from.

    Steps are stored in dependency order (every antecedent appears before the belief it
    supports), so ``proof.steps[-1]`` is a root and ``proof.premises`` are the leaves. A proof is
    a snapshot: it records the values, sources and justifications at the moment it was built, so
    it can be exported, diffed and verified independently of the belief base.
    """

    roots: tuple[str, ...]
    steps: tuple[ProofStep, ...]
    created_at: datetime = field(default_factory=utcnow)

    # -- construction ---------------------------------------------------------------------

    @classmethod
    def build(cls, kb: BeliefBase, *keys_or_refs: str) -> Proof:
        """Snapshot the support of ``keys_or_refs`` in ``kb`` (see :meth:`BeliefBase.proof`)."""
        roots = tuple(kb._resolve_for_proof(item) for item in keys_or_refs)
        ordered: list[ProofStep] = []
        seen: set[str] = set()

        # Iterative post-order walk over support edges; supports are acyclic by construction.
        stack: list[tuple[str, bool]] = [(ref, False) for ref in reversed(roots)]
        while stack:
            ref, expanded = stack.pop()
            if ref in seen:
                continue
            just = kb._proof_justification(ref)
            if expanded:
                seen.add(ref)
                ordered.append(
                    ProofStep(
                        belief=kb._nodes[ref].belief,
                        status=kb._nodes[ref].status,
                        confidence=kb.confidence(ref),
                        justification=just,
                    )
                )
                continue
            stack.append((ref, True))
            if just is not None:
                for antecedent in reversed(just.antecedents):
                    if antecedent not in seen:
                        stack.append((antecedent, False))
        return cls(roots=roots, steps=tuple(ordered), created_at=kb.now())

    # -- access ---------------------------------------------------------------------------

    def __iter__(self) -> Iterator[ProofStep]:
        return iter(self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    def __contains__(self, key_or_ref: object) -> bool:
        return any(key_or_ref in (s.belief.key, s.ref) for s in self.steps)

    def step(self, key_or_ref: str) -> ProofStep:
        for s in self.steps:
            if key_or_ref in (s.ref, s.belief.key):
                return s
        raise KeyError(key_or_ref)

    @property
    def by_ref(self) -> dict[str, ProofStep]:
        return {s.ref: s for s in self.steps}

    @property
    def premises(self) -> list[ProofStep]:
        """The leaves: beliefs grounded directly in a source."""
        return [s for s in self.steps if s.is_premise]

    @property
    def derived(self) -> list[ProofStep]:
        return [s for s in self.steps if not s.is_premise]

    @property
    def valid(self) -> bool:
        """True when every step was ``IN`` at the time the proof was taken."""
        return all(s.status is Status.IN for s in self.steps)

    # -- verification ---------------------------------------------------------------------

    def verify(
        self,
        kb: BeliefBase | None = None,
        *,
        checks: list[Check] | None = None,
        at: datetime | None = None,
    ) -> VerificationReport:
        """Run the deterministic verifier over this proof. See :mod:`corollary.verify`."""
        from .verify import Verifier

        return Verifier(checks).verify(self, kb=kb, at=at)

    # -- rendering ------------------------------------------------------------------------

    def render(self, *, show_sources: bool = True, ascii: bool = False) -> str:
        """Render the proof as a tree, roots first.

        ::

            growth:Q3_vs_Q2 = 9.7561  [IN 0.99]  rule:growth
            ├── revenue:Q2 = 4,100,000,000  [IN 0.99]  tool:get_revenue(quarter='Q2')
            └── revenue:Q3 = 4,500,000,000  [IN 0.99]  tool:get_revenue(quarter='Q3')

        Pass ``ascii=True`` for consoles that cannot display box-drawing characters.
        """
        tee, elbow, pipe = ("|-- ", "`-- ", "|   ") if ascii else ("├── ", "└── ", "│   ")
        by_ref = self.by_ref
        lines: list[str] = []
        printed: set[str] = set()

        def label(step: ProofStep) -> str:
            text = f"{step.belief.key} = {format_value(step.belief.value)}  [{step.status.value} {step.confidence:.2f}]"
            if show_sources:
                text += f"  {step.justification.describe() if step.justification else step.belief.source}"
            return text

        # Depth-first with an explicit stack, so a proof thousands of steps deep can still render.
        stack: list[tuple[str, str, str, str]] = [(root, "", "", "") for root in reversed(self.roots)]
        while stack:
            ref, prefix, connector, child_prefix = stack.pop()
            step = by_ref[ref]
            again = ref in printed
            lines.append(f"{prefix}{connector}{label(step)}{'  (see above)' if again and step.antecedents else ''}")
            if again:
                continue
            printed.add(ref)
            children = [a for a in step.antecedents if a in by_ref]
            for i in reversed(range(len(children))):
                last = i == len(children) - 1
                stack.append((children[i], prefix + child_prefix, elbow if last else tee, "    " if last else pipe))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render(ascii=not _stdout_supports_unicode())

    def to_mermaid(self) -> str:
        """Mermaid flowchart source (arrows point from antecedent to conclusion)."""
        ids = {s.ref: f"n{i}" for i, s in enumerate(self.steps)}
        lines = ["flowchart BT"]
        for s in self.steps:
            text = _mermaid_text(f"{s.belief.key} = {format_value(s.belief.value)}")
            shape = ('(["', '"])') if s.is_premise else ('["', '"]')
            lines.append(f"    {ids[s.ref]}{shape[0]}{text}{shape[1]}")
        for s in self.steps:
            edge_label = _mermaid_text(s.justification.describe()) if s.justification else ""
            for a in s.antecedents:
                if a in ids:
                    lines.append(f'    {ids[a]} -->|"{edge_label}"| {ids[s.ref]}')
        return "\n".join(lines)

    def to_dot(self) -> str:
        """Graphviz DOT source."""
        ids = {s.ref: f"n{i}" for i, s in enumerate(self.steps)}
        lines = ["digraph proof {", "  rankdir=BT;", "  node [shape=box, fontname=Helvetica];"]
        for s in self.steps:
            text = _dot_text(f"{s.belief.key} = {format_value(s.belief.value)}")
            style = ", style=rounded" if s.is_premise else ""
            lines.append(f'  {ids[s.ref]} [label="{text}"{style}];')
        for s in self.steps:
            for a in s.antecedents:
                if a in ids:
                    lines.append(f"  {ids[a]} -> {ids[s.ref]};")
        lines.append("}")
        return "\n".join(lines)

    # -- comparison and serialization ---------------------------------------------------

    def diff(self, other: Proof) -> ProofDiff:
        """Compare by key: beliefs present only in ``other`` are ``added``."""
        mine = {s.belief.key: s for s in self.steps}
        theirs = {s.belief.key: s for s in other.steps}
        from .belief import values_equal

        both = [k for k in mine if k in theirs]
        return ProofDiff(
            added=tuple(k for k in theirs if k not in mine),
            removed=tuple(k for k in mine if k not in theirs),
            changed=tuple(
                (k, mine[k].belief.value, theirs[k].belief.value)
                for k in both
                if not values_equal(mine[k].belief.value, theirs[k].belief.value)
            ),
            status_changed=tuple(
                (k, mine[k].status, theirs[k].status) for k in both if mine[k].status is not theirs[k].status
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "corollary.proof",
            "version": 1,
            "roots": list(self.roots),
            "created_at": self.created_at.isoformat(),
            "steps": [s.to_dict() for s in self.steps],
        }

    def to_json(self, **kwargs: Any) -> str:
        """The proof as JSON. Keyword arguments go to :func:`json.dumps`. A value JSON can't represent
        (a ``Decimal``, a ``date``) raises ``TypeError`` rather than being silently turned into a
        string; pass ``default=str`` to accept that loss."""
        kwargs.setdefault("indent", 2)
        try:
            return json.dumps(self.to_dict(), **kwargs)
        except TypeError as exc:
            if "default" in kwargs:
                raise
            for step in self.steps:
                try:
                    json.dumps(step.belief.value, **kwargs)
                except TypeError:
                    value_type = type(step.belief.value).__name__
                    raise TypeError(
                        f"belief {step.ref} has a value of type {value_type}, which JSON can't represent; "
                        "pass default=str to store it as text"
                    ) from exc
            raise

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Proof:
        if data.get("format") != "corollary.proof":
            raise ValueError(f"not a Corollary proof (format {data.get('format')!r})")
        version = data.get("version")
        if isinstance(version, bool) or not isinstance(version, int) or version != 1:
            raise ValueError(f"unsupported proof version {version!r}; upgrade corollary to read it")
        return cls(
            roots=tuple(data["roots"]),
            steps=tuple(ProofStep.from_dict(s) for s in data["steps"]),
            created_at=datetime.fromisoformat(data["created_at"]),
        )

    @classmethod
    def from_json(cls, text: str) -> Proof:
        return cls.from_dict(json.loads(text))


def _mermaid_text(text: str) -> str:
    """Escape text for a quoted Mermaid label: entity codes survive where raw quotes would not."""
    return text.replace("#", "#35;").replace('"', "#quot;").replace("\n", " ")


def _dot_text(text: str) -> str:
    """Escape text for a quoted DOT string: backslashes first, then quotes."""
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _stdout_supports_unicode() -> bool:
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "├└│".encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True
