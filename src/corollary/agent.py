"""The agent runtime: a stateless model proposing beliefs to a kernel that owns the state.

Each step, the projector renders the ``IN`` beliefs into a context, the model answers in the
claim contract, and every action is validated before the belief base changes:

* tool calls are executed by the runtime, which records the results as premises;
* citations must quote the document and state the cited value;
* claims may depend only on beliefs that were visible in that step's context, and formulas
  must reproduce the stated value;
* the final answer is itself a belief, so it goes ``OUT`` when its support is retracted, and
  :meth:`Agent.repair` re-derives it.
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .belief import Belief, Source, SourceKind, Status, format_value, parse_ref, validate_key, values_equal
from .changes import Change, Propagation
from .conflict import Resolver
from .contract import SYSTEM_PROMPT, Action, Answer, Cite, Claim, ToolCall, parse_response
from .errors import CitationError, ContractViolation, CorollaryError, FormulaError, ModelError
from .formula import evaluate, formula_keys
from .justification import Justification, JustificationKind
from .kernel import BeliefBase, Derived, Rederivation
from .models.base import Model, resolve_model
from .projector import Projection, Projector
from .proof import Proof
from .rules import Rule
from .textmatch import normalize
from .tools import Tool, tool
from .trust import TrustPolicy
from .verify import Check, VerificationReport

_ANSWER_NOTE = "Answer the task: "

log = logging.getLogger(__name__)


class Dependencies(str, enum.Enum):
    """How the runtime decides what a model claim depends on."""

    CONSERVATIVE = "conservative"
    """A claim depends on every belief in the context it was generated from. Sound by
    construction: a retracted fact can never survive in a conclusion. May over-retract."""
    DECLARED = "declared"
    """A claim depends on the keys the model lists in ``follows_from`` (plus formula keys).
    Sharper cascades, but trusts the model's account of what it used."""

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class StepRecord:
    """What happened in one model call: the context, the raw response, and the verdicts."""

    index: int
    prompt: str
    response: str
    accepted: tuple[str, ...] = ()
    rejected: tuple[str, ...] = ()


@dataclass
class Report:
    """The outcome of :meth:`Agent.run`. Reads the belief base live, so after a retraction and
    :meth:`Agent.repair`, ``report.answer`` is the repaired answer."""

    task: str
    key: str
    kb: BeliefBase
    completed: bool
    steps: tuple[StepRecord, ...] = ()
    changes: tuple[Change, ...] = ()
    """Everything that became ``IN`` or ``OUT`` during the run."""
    error: ModelError | None = None
    """The model error that ended the run early (a refusal, a truncated response, an API failure).
    Everything the run established before it stays in the belief base and in ``steps``."""

    @property
    def belief(self) -> Belief | None:
        """The currently believed answer, or ``None`` if there is none (not answered, or ``OUT``)."""
        if self.key not in self.kb.keys(status=None):
            return None
        result = self.kb.get(self.key)
        return result if isinstance(result, Belief) else None

    @property
    def answer(self) -> str | None:
        belief = self.belief
        return str(belief.value) if belief is not None else None

    @property
    def stale(self) -> bool:
        """True when the agent answered but the answer has since lost its support."""
        return self.completed and self.kb.status(self.key) is Status.OUT

    @property
    def proof(self) -> Proof:
        """The graph of beliefs the answer follows from."""
        if self.key not in self.kb.keys(status=None):
            raise CorollaryError("the agent did not produce an answer, so there is no proof")
        return self.kb.proof(self.key)

    def verify(self, checks: list[Check] | None = None, *, raise_on_error: bool = False) -> VerificationReport:
        """Deterministically check the proof. See :mod:`corollary.verify`."""
        report = self.proof.verify(self.kb, checks=checks)
        return report.raise_for_errors() if raise_on_error else report

    @property
    def rejections(self) -> list[str]:
        return [r for s in self.steps for r in s.rejected]

    def __str__(self) -> str:
        return self.answer if self.answer is not None else "(no answer)"

    def __repr__(self) -> str:
        def short(value: str | None) -> str:
            if value is not None and len(value) > 60:
                value = value[:57] + "..."
            return repr(value)

        return (
            f"Report(task={short(self.task)}, completed={self.completed!r}, answer={short(self.answer)}, "
            f"steps={len(self.steps)}, rejections={len(self.rejections)}, error={self.error!r})"
        )


@dataclass(frozen=True)
class NarrowResult:
    """Result of :meth:`Agent.narrow`: which dependencies were needed and which were pruned."""

    key: str
    kept: tuple[str, ...]
    pruned: tuple[str, ...]
    calls: int


@dataclass
class _StepState:
    projection: Projection
    claimed: dict[str, Belief] = field(default_factory=dict)
    tool_keys: set[str] = field(default_factory=set)


class Agent:
    """An agent whose state is a belief base, not a message log.

    Args:
        model: A :class:`~corollary.models.Model`, or a string such as ``"anthropic:claude-opus-5-5"``.
        beliefs: The belief base to read and write. A new one is created when omitted.
        tools: Tools (or plain functions) the model may ask the runtime to call.
        documents: ``{name: text}`` the model may cite; quotes are span-checked.
        rules: Deterministic rules to register on the belief base.
        trust: Trust policy for a newly created belief base (or to replace the given one's).
        projector: Builds model contexts; see :class:`Projector`.
        dependencies: ``"conservative"`` (default) or ``"declared"``; see :class:`Dependencies`.
        resolver: Applied to open conflicts before every step; see :mod:`corollary.resolvers`.
        max_steps: Model calls allowed per :meth:`run`.
        repair_attempts: Model calls allowed per belief during :meth:`repair`.
        self_consistency: Ask the model ``k`` times for every unverified claim (no formula) and use
            the agreement rate as that step's certainty. ``1`` (default) disables it; each extra
            sample is one model call.
        learn_from_checks: Record the model's verified work (formulas and citations that check
            out, or don't) in the trust ledger, so its reliability is measured rather than assumed.
        instructions: Domain guidance shown to the model on every step, under the task (house
            style, what to answer in which language, ...). The claim contract stays in force.
        system_prompt: The contract the model is held to. Replace it only to adapt the wording.
    """

    def __init__(
        self,
        model: Model | str,
        beliefs: BeliefBase | None = None,
        tools: Iterable[Tool | Callable[..., Any]] = (),
        *,
        documents: Mapping[str, str] | None = None,
        rules: Iterable[Rule] = (),
        trust: TrustPolicy | None = None,
        projector: Projector | None = None,
        dependencies: Dependencies | str = Dependencies.CONSERVATIVE,
        resolver: Resolver | None = None,
        max_steps: int = 12,
        repair_attempts: int = 2,
        self_consistency: int = 1,
        learn_from_checks: bool = True,
        instructions: str = "",
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        if self_consistency < 1:
            raise ValueError("self_consistency must be at least 1")
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.model = resolve_model(model)
        self.kb = beliefs if beliefs is not None else BeliefBase(trust=trust)
        if trust is not None:
            self.kb.trust = trust
        self.tools: dict[str, Tool] = {}
        for t in tools:
            t = t if isinstance(t, Tool) else tool(t)
            self.tools[t.name] = t
        for name, text in (documents or {}).items():
            self.kb.add_document(name, text)
        for r in rules:
            self.kb.register_rule(r)
        self.projector = projector or Projector()
        self.dependencies = Dependencies(dependencies)
        self.resolver = resolver
        self.max_steps = max_steps
        self.repair_attempts = repair_attempts
        self.self_consistency = self_consistency
        self.learn_from_checks = learn_from_checks
        self.instructions = instructions
        self.system_prompt = system_prompt

    @property
    def beliefs(self) -> BeliefBase:
        return self.kb

    # ======================================================================================
    # Running a task
    # ======================================================================================

    def run(self, task: str, *, max_steps: int | None = None, instructions: str = "") -> Report:
        """Work on ``task`` until the model answers or the step budget runs out.

        ``instructions`` are added to the agent's own for this run only.
        """
        guidance = "\n\n".join(part.strip() for part in (self.instructions, instructions) if part.strip())
        key = self._answer_key()
        steps: list[StepRecord] = []
        feedback: list[str] = []
        budget = self.max_steps if max_steps is None else max_steps
        if budget < 1:
            raise ValueError("max_steps must be at least 1")
        for index in range(1, budget + 1):
            self.kb.refresh()
            if self.resolver is not None:
                self.kb.resolve_conflicts(self.resolver)
            projection = self.projector.project(
                self.kb, task=task, tools=list(self.tools.values()), feedback=feedback, instructions=guidance
            )
            try:
                response = self.model.complete(self.system_prompt, projection.text)
            except ModelError as exc:
                log.warning("model %s failed on step %d of %r: %s", self.model.name, index, task, exc)
                steps.append(StepRecord(index, projection.text, "", (), (f"model error: {exc}",)))
                return Report(task, key, self.kb, False, tuple(steps), tuple(self.kb.changes()), error=exc)
            accepted, rejected, answered = self._process(response, projection, key, task)
            log.debug("step %d: accepted %s, rejected %s", index, accepted, rejected)
            steps.append(StepRecord(index, projection.text, response, tuple(accepted), tuple(rejected)))
            if answered:
                return Report(task, key, self.kb, True, tuple(steps), tuple(self.kb.changes()))
            feedback = rejected or ([] if accepted else ["the response contained no actions"])
        return Report(task, key, self.kb, False, tuple(steps), tuple(self.kb.changes()))

    def _process(self, response: str, projection: Projection, key: str, task: str) -> tuple[list[str], list[str], bool]:
        accepted: list[str] = []
        rejected: list[str] = []
        try:
            parsed = parse_response(response)
        except ContractViolation as exc:
            return accepted, list(exc.errors), False
        rejected.extend(parsed.errors)
        state = _StepState(projection)
        for action in parsed.actions:
            try:
                accepted.append(self._apply(action, state, key, task))
            except CorollaryError as exc:
                rejected.append(f"{_describe(action)} rejected: {exc}")
                continue
            if isinstance(action, Answer):
                return accepted, rejected, True
        return accepted, rejected, False

    def _apply(self, action: Action, state: _StepState, answer_key: str, task: str) -> str:
        if isinstance(action, ToolCall):
            return self._apply_tool_call(action, state, answer_key)
        if isinstance(action, (Cite, Claim)) and action.key == answer_key:
            raise ContractViolation(f"{answer_key!r} is reserved for the answer")
        if isinstance(action, Cite):
            try:
                belief = self.kb.cite(
                    validate_key(action.key),
                    action.value,
                    document=action.document,
                    quote=action.quote,
                    claim=action.claim,
                )
            except CitationError as exc:
                if not str(exc).startswith("unknown document"):
                    self._learn(False, f"citation rejected: {exc}")
                raise
            self._learn(True, f"citation of {action.document!r} verified")
            return f"cited {belief.ref} from {action.document!r}"
        if isinstance(action, Claim):
            belief = self._apply_claim(
                action.key,
                action.value,
                action.claim,
                action.follows_from,
                action.formula,
                action.confidence,
                state,
                note=action.claim or action.key,
            )
            state.claimed[belief.key] = belief
            return f"claimed {belief.ref} = {format_value(belief.value)}"
        belief = self._apply_claim(
            answer_key,
            action.text,
            action.text,
            action.follows_from,
            None,
            action.confidence,
            state,
            note=_ANSWER_NOTE + task,
            is_answer=True,
        )
        return f"answered as {belief.ref}"

    def _apply_tool_call(self, action: ToolCall, state: _StepState, answer_key: str) -> str:
        t = self.tools.get(action.tool)
        if t is None:
            available = ", ".join(self.tools) or "none"
            raise ContractViolation(f"unknown tool {action.tool!r} (available: {available})")
        args = t.bind(action.args)
        key = validate_key(action.key or t.default_key(args))
        if key == answer_key:
            raise ContractViolation(f"{answer_key!r} is reserved for the answer")
        try:
            value = t.invoke(args)
        except Exception as exc:
            log.warning("tool %r raised for arguments %r", t.name, args, exc_info=True)
            raise ContractViolation(f"tool {t.name!r} raised {type(exc).__name__}: {exc}") from exc
        belief = self._record_tool_result(t, args, key, value, action.claim)
        state.tool_keys.add(key)
        return f"called {t.name} -> {belief.ref} = {format_value(value)}"

    def _record_tool_result(self, t: Tool, args: Mapping[str, Any], key: str, value: Any, claim: str) -> Belief:
        source = Source.tool(t.name, args, origin=t.origin)
        # A fresh result from the same call replaces the old observation instead of conflicting with it.
        supersede = any(b.source == source for b in self.kb.beliefs(Status.IN) if b.key == key)
        return self.kb.assert_(
            key,
            value,
            source=source,
            claim=claim,
            confidence=self.kb.trust.tool_confidence(t.trust),
            ttl=t.ttl,
            half_life=t.half_life,
            supersede=supersede,
        )

    def _apply_claim(
        self,
        key: str,
        value: Any,
        claim: str,
        follows_from: Sequence[str],
        formula: str | None,
        confidence: float | None,
        state: _StepState,
        *,
        note: str,
        is_answer: bool = False,
    ) -> Belief:
        validate_key(key)
        deps = _dependencies(follows_from, formula)
        for dep in deps:
            self._check_visible(dep, state)
        if formula is not None:
            values = {d: self._visible_value(d, state) for d in formula_keys(formula)}
            value = self._checked_formula_value(key, formula, value, values)
        if value is None:
            raise ContractViolation(f"claim {key!r} has no value")
        existing = self.kb.get(key)
        if isinstance(existing, Belief) and not values_equal(existing.value, value):
            raise ContractViolation(
                f"{key!r} already holds {format_value(existing.value)}; use a new key for a different value"
            )
        if self.dependencies is Dependencies.DECLARED:
            if is_answer and not deps:
                raise ContractViolation("an answer must list the beliefs it rests on in follows_from")
            antecedents = [self._visible_belief(d, state) for d in deps]
        else:
            visible = list(state.projection.visible) + list(state.claimed.values())
            gone = [b.ref for b in visible if self.kb.status(b.ref) is not Status.IN]
            if gone:
                raise ContractViolation(
                    f"beliefs in this turn's context changed during the turn ({', '.join(gone)}); "
                    "derive it again next turn"
                )
            antecedents = visible
        certainty = self._certainty(confidence)
        if self.self_consistency > 1 and formula is None and not is_answer:
            certainty *= self._consistency(key, value, note, [b.key for b in antecedents])
        return self.kb.justify(
            key,
            value,
            antecedents=[b.ref for b in antecedents],
            source=Source.model(self.model.name),
            claim=claim,
            formula=formula,
            confidence=certainty,
            inputs=[b.key for b in antecedents],
            note=note,
        )

    def _checked_formula_value(self, key: str, formula: str, value: Any, values: Mapping[str, Any]) -> Any:
        """Re-execute ``formula``: return the computed value when none was stated, the stated value
        when the formula reproduces it, and reject the claim otherwise. Either way the model's
        track record learns from it."""
        computed = evaluate(formula, values)
        if value is None:
            return computed
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not values_equal(computed, value, rel_tol=1e-6, abs_tol=1e-9)
        ):
            self._learn(False, f"formula for {key!r} did not reproduce the stated value")
            raise ContractViolation(
                f"formula {formula!r} evaluates to {format_value(computed)}, but the claim states {format_value(value)}"
            )
        self._learn(True, f"formula for {key!r} verified")
        return value

    @staticmethod
    def _certainty(stated: float | None) -> float:
        """The step's own certainty. The model's reliability (learned in the trust ledger) and
        rule-level trust for re-executed formulas are applied by the belief base on top."""
        return stated if stated is not None else 1.0

    def _consistency(self, key: str, value: Any, note: str, scope: list[str]) -> float:
        """Ask for the same claim ``k - 1`` more times; return ``(agreeing + 1) / (k + 1)``.

        Every sample sees the same beliefs but not the original answer. A claim the model
        reproduces every time keeps its certainty; one it can't reproduce loses most of it.
        """
        k = self.self_consistency
        agreeing = 1  # the original answer
        for _ in range(k - 1):
            sample = self._ask(key, note, scope, mode="check")
            if sample is not None and _same_value(sample.value, value):
                agreeing += 1
        return (agreeing + 1) / (k + 1)

    def _learn(self, correct: bool, reason: str) -> None:
        if self.learn_from_checks:
            self.kb.record_outcome(Source.model(self.model.name), correct, reason=reason)

    def _check_visible(self, key: str, state: _StepState) -> None:
        if key in state.claimed or state.projection.belief(key) is not None:
            return
        if key in state.tool_keys:
            raise ContractViolation(
                f"depends on {key!r}, the result of a tool called in this same response; use it next turn"
            )
        if key in self.kb.keys(status=None):
            raise ContractViolation(
                f"depends on {key!r}, which is not in this turn's context (OUT, conflicted or hidden)"
            )
        raise ContractViolation(f"depends on unknown key {key!r}")

    def _visible_belief(self, key: str, state: _StepState) -> Belief:
        belief = state.claimed.get(key) or state.projection.belief(key)
        assert belief is not None  # guarded by _check_visible
        return belief

    def _visible_value(self, key: str, state: _StepState) -> Any:
        try:
            self._check_visible(key, state)
        except ContractViolation as exc:
            raise FormulaError(str(exc)) from None
        return self._visible_belief(key, state).value

    def _answer_key(self) -> str:
        existing = set(self.kb.keys(status=None))
        if "answer" not in existing:
            return "answer"
        n = 2
        while f"answer:{n}" in existing:
            n += 1
        return f"answer:{n}"

    # ======================================================================================
    # Repair, re-verification and narrowing
    # ======================================================================================

    def repair(self, *, include_kept: bool = False) -> Propagation:
        """Re-derive everything that lost support: rules directly, model claims through the model.

        Call this after correcting an input (``kb.retract`` + ``kb.assert_``). The model sees
        only the current values of each belief's inputs, never the retracted ones.
        """
        return self.kb.propagate(rederive=self._rederive, include_kept=include_kept)

    def reverify(self, *, include_kept: bool = False) -> Propagation:
        """Re-run the tool calls behind expired or faded premises, then :meth:`repair`.

        Expired premises are ``OUT`` (see ``ttl``); faded ones are still ``IN`` but their decaying
        confidence has dropped below the trust policy's ``min_confidence`` (see ``half_life``). A
        refreshed result equal to the old one renews the evidence in place, so nothing downstream
        needs to be re-derived.
        """
        for belief in [*self.kb.stale(), *self.kb.faded()]:
            just = _latest_tool_premise(self.kb.justifications(belief.ref))
            if just is None or just.source is None:
                continue
            t = self.tools.get(just.source.name)
            if t is None:
                continue
            args = just.source.args
            try:
                value = t.invoke(args)
            except Exception:
                # Stays stale; the next reverify() will try again.
                log.warning("re-running tool %r for %s failed", t.name, belief.ref, exc_info=True)
                continue
            self._record_tool_result(t, args, belief.key, value, belief.claim)
        return self.repair(include_kept=include_kept)

    def narrow(self, key: str) -> NarrowResult:
        """Prune unnecessary dependencies of a model-derived belief by ablation.

        For each antecedent in turn, the model is asked to derive ``key`` again with that belief
        removed from its context. If the value is unchanged, the belief was not needed. The
        result is recorded as an additional, narrower justification, so later retractions of a
        pruned belief no longer cascade into ``key``. Costs one model call per antecedent.
        """
        belief = self.kb[key]
        just = self.kb.support(belief.ref)
        if just is None or just.kind is not JustificationKind.MODEL:
            raise CorollaryError(f"{key!r} is not supported by a model justification; nothing to narrow")
        current = list(dict.fromkeys(parse_ref(a)[0] for a in just.antecedents))
        pruned: list[str] = []
        calls = 0
        for candidate in list(current):
            trial = [k for k in current if k != candidate]
            calls += 1
            derived = self._ask(key, just.note or belief.claim, trial, mode="ablate")
            if derived is not None and _same_value(derived.value, belief.value):
                current, pruned = trial, [*pruned, candidate]
        if pruned:
            formula = just.formula if just.formula and set(formula_keys(just.formula)) <= set(current) else None
            self.kb.justify(
                key,
                belief.value,
                antecedents=[self.kb[k].ref for k in current],
                source=just.source or Source.model(self.model.name),
                claim=belief.claim,
                formula=formula,
                confidence=just.confidence,
                inputs=current,
                note=just.note,
            )
        return NarrowResult(key, tuple(current), tuple(pruned), calls)

    def _rederive(self, request: Rederivation) -> Derived | None:
        if not request.inputs:
            return None
        belief, just = request.belief, request.justification
        return self._ask(
            belief.key, just.note or belief.claim, list(request.inputs), mode="rederive", previous=belief.value
        )

    def _ask(self, key: str, note: str, scope: list[str], *, mode: str, previous: Any = None) -> Derived | None:
        """Ask the model for one claim about ``key`` using only the beliefs in ``scope``.

        ``mode`` is ``"rederive"`` (inputs changed; the previous value is mentioned), or
        ``"ablate"`` / ``"check"`` (derive from scratch, without seeing any previous value).
        """
        is_answer = note.startswith(_ANSWER_NOTE)
        if mode == "rederive":
            intro = (
                f"Re-derive the belief `{key}`. It previously held {format_value(previous)}, but the "
                "beliefs it was derived from have changed."
            )
        else:
            intro = (
                f"Derive the belief `{key}` using only the beliefs below. If they are not sufficient, "
                'respond with {"actions": []}.'
            )
        what = "the complete, updated answer text" if is_answer else "its value"
        if is_answer:
            subject = note  # the task, which doesn't contain the answer
        elif mode == "rederive":
            subject = f"`{key}`: {note}"
        else:
            # Deriving from scratch: the claim text states the old conclusion, so showing it would
            # let the model copy it, and every ablation or consistency check would trivially agree.
            subject = f"The belief to derive: `{key}`."
        lines = [
            subject,
            intro,
            f'Respond with exactly one action of type "claim" for key `{key}` whose "value" is {what}. '
            "Use only the beliefs listed below, and include a formula if the value is computed.",
        ]
        task = "\n\n".join(lines)
        feedback: list[str] = []
        for _ in range(max(1, self.repair_attempts)):
            projection = self.projector.project(
                self.kb,
                task=task,
                scope=scope,
                feedback=feedback,
                instructions=self.instructions,
                include_documents=False,
            )
            try:
                response = self.model.complete(self.system_prompt, projection.text)
                parsed = parse_response(response)
            except (ContractViolation, ModelError) as exc:
                feedback = [str(exc)]
                continue
            claims = [a for a in parsed.actions if isinstance(a, Claim) and a.key == key]
            if not parsed.actions and not parsed.errors:
                return None  # The model says the beliefs are not sufficient.
            if not claims and parsed.errors:
                feedback = list(parsed.errors)
                continue
            if not claims:
                feedback = [f'respond with one "claim" action for key `{key}`']
                continue
            claim = claims[0]
            try:
                return self._validate_rederived(claim, projection, is_answer)
            except CorollaryError as exc:
                feedback = [str(exc)]
        return None

    def _validate_rederived(self, claim: Claim, projection: Projection, is_answer: bool) -> Derived:
        allowed = set(projection.keys)
        deps = _dependencies(claim.follows_from, claim.formula)
        outside = [d for d in deps if d not in allowed]
        if outside:
            raise ContractViolation(f"uses beliefs outside the provided context: {', '.join(outside)}")
        value = claim.value
        if claim.formula:
            values = {k: _required(projection.belief(k)).value for k in formula_keys(claim.formula)}
            value = self._checked_formula_value(claim.key, claim.formula, value, values)
        if value is None:
            raise ContractViolation("the claim has no value")
        return Derived(
            value=value,
            claim=str(value) if is_answer else claim.claim,
            formula=claim.formula,
            confidence=self._certainty(claim.confidence),
            antecedents=tuple(deps) if self.dependencies is Dependencies.DECLARED and deps else None,
        )


def _describe(action: Action) -> str:
    if isinstance(action, ToolCall):
        return f"call_tool {action.tool!r}"
    if isinstance(action, Cite):
        return f"cite {action.key!r}"
    if isinstance(action, Claim):
        return f"claim {action.key!r}"
    return "answer"


def _dependencies(follows_from: Sequence[str], formula: str | None) -> list[str]:
    """The keys a claim rests on: the ones it lists, plus the ones its formula reads."""
    return list(dict.fromkeys([*follows_from, *(formula_keys(formula) if formula else [])]))


def _required(belief: Belief | None) -> Belief:
    assert belief is not None  # formula keys were checked against the context first
    return belief


def _latest_tool_premise(justifications: Sequence[Justification]) -> Justification | None:
    for j in reversed(justifications):
        if j.is_premise and j.source is not None and j.source.kind is SourceKind.TOOL:
            return j
    return None


def _same_value(a: Any, b: Any) -> bool:
    if isinstance(a, str) and isinstance(b, str):
        return normalize(a) == normalize(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return values_equal(a, b, rel_tol=1e-6, abs_tol=1e-9)
    return values_equal(a, b)
