from __future__ import annotations

import pytest

from corollary import (
    BeliefBase,
    ChangeKind,
    CircularDefeatError,
    CitationError,
    FormulaError,
    JustificationKind,
    NotBelievedError,
    RuleError,
    Source,
    Status,
    UnknownBeliefError,
    UnresolvedConflictError,
    rule,
)


@rule
def growth(previous: float, current: float) -> float:
    return (current - previous) / previous * 100


@rule
def trend(g: float) -> str:
    return "strong" if g >= 8 else "modest" if g >= 3 else "flat"


def kinds(changes: object) -> list[tuple[str, str]]:
    return [(c.kind.value, c.key) for c in changes]  # type: ignore[attr-defined]


# -- asserting --------------------------------------------------------------------------------


def test_assert_creates_in_premise(kb: BeliefBase) -> None:
    b = kb.assert_("revenue:Q2", 4.3e9, source="tool:get_revenue", claim="Q2 revenue")
    assert b.ref == "revenue:Q2@1"
    assert kb.status("revenue:Q2") is Status.IN
    assert kb.value("revenue:Q2") == 4.3e9
    assert "revenue:Q2" in kb and "revenue:Q2@1" in kb
    assert kb.support("revenue:Q2").kind is JustificationKind.PREMISE  # type: ignore[union-attr]
    assert kb.confidence("revenue:Q2") == pytest.approx(0.95)


def test_same_value_corroborates_instead_of_new_revision(kb: BeliefBase) -> None:
    kb.assert_("x", 1, source="tool:a")
    b = kb.assert_("x", 1, source="human:alice")
    assert b.revision == 1
    assert len(kb.justifications("x")) == 2
    assert len(kb.revisions("x")) == 1


def test_different_value_creates_revision_and_conflict(kb: BeliefBase) -> None:
    kb.assert_("x", 1, source="tool:a")
    b = kb.assert_("x", 2, source="tool:b")
    assert b.ref == "x@2"
    (conflict,) = kb.conflicts()
    assert conflict.subject == "x" and set(conflict.refs) == {"x@1", "x@2"}
    assert len(conflict.proofs) == 2
    with pytest.raises(UnresolvedConflictError):
        kb.value("x")
    with pytest.raises(UnresolvedConflictError):
        kb.derive("y", rule(lambda v: v, name="ident"), "x")


def test_supersede_retracts_old_revision(kb: BeliefBase) -> None:
    kb.assert_("price", 10, source="tool:quote")
    kb.assert_("price", 11, source="tool:quote", supersede=True)
    assert kb.value("price") == 11
    assert kb.status("price@1") is Status.OUT
    assert "superseded" in (kb.why_out("price@1") or "")
    assert kb.conflicts() == []


def test_rule_sources_cannot_be_asserted(kb: BeliefBase) -> None:
    with pytest.raises(ValueError, match="reserved"):
        kb.assert_("x", 1, source="rule:foo")


def test_confidence_validation(kb: BeliefBase) -> None:
    with pytest.raises(ValueError):
        kb.assert_("x", 1, confidence=1.5)


def test_get_default_and_unknown(kb: BeliefBase) -> None:
    assert kb.get("nope") is None
    assert kb.get("nope", 5) == 5
    with pytest.raises(UnknownBeliefError):
        kb.status("nope")
    with pytest.raises(UnknownBeliefError):
        _ = kb["nope"]
    assert "nope" not in kb
    assert 42 not in kb


# -- deriving -------------------------------------------------------------------------------


def test_derive_with_rule(revenue_kb: BeliefBase) -> None:
    kb = revenue_kb
    assert kb.value("growth:Q3_vs_Q2") == pytest.approx(4.651162790697675)
    assert kb.value("trend:Q3") == "modest"
    support = kb.support("growth:Q3_vs_Q2")
    assert support is not None and support.rule == "growth"
    assert support.antecedents == ("revenue:Q2@1", "revenue:Q3@1")
    assert support.inputs == ("revenue:Q2", "revenue:Q3")
    assert [b.key for b in kb.dependents("revenue:Q2")] == ["growth:Q3_vs_Q2", "trend:Q3"]
    assert [b.key for b in kb.dependents("revenue:Q2", transitive=False)] == ["growth:Q3_vs_Q2"]


def test_derive_requires_believed_inputs(kb: BeliefBase) -> None:
    kb.assert_("a", 1)
    kb.retract("a")
    with pytest.raises(NotBelievedError):
        kb.derive("b", rule(lambda v: v, name="ident"), "a")
    with pytest.raises(NotBelievedError):
        kb.derive("b", "ident", "a@1")


def test_derive_wraps_rule_errors(kb: BeliefBase) -> None:
    kb.assert_("a", 0)
    with pytest.raises(RuleError, match="failed"):
        kb.derive("b", rule(lambda v: 1 / v, name="inv"), "a")


def test_rules_by_name_and_conflicting_registration(kb: BeliefBase) -> None:
    kb.register_rule(growth)
    kb.assert_("a", 1.0)
    kb.assert_("b", 2.0)
    assert kb.derive("g", "growth", "a", "b").value == 100
    with pytest.raises(RuleError):
        kb.register_rule(rule(lambda a, b: 0, name="growth"))
    with pytest.raises(RuleError):
        kb.derive("h", "missing_rule", "a")


def test_plain_callables_become_rules(kb: BeliefBase) -> None:
    def double(x: float) -> float:
        return x * 2

    kb.assert_("a", 2)
    assert kb.derive("b", double, "a").value == 4
    assert "double" in kb.rules


def test_justify_checks_formula(kb: BeliefBase) -> None:
    kb.assert_("a", 2)
    kb.assert_("b", 3)
    ok = kb.justify("c", 6, antecedents=["a", "b"], source="model:m", formula="{a} * {b}")
    assert ok.value == 6
    with pytest.raises(FormulaError, match="evaluates to"):
        kb.justify("d", 7, antecedents=["a", "b"], source="model:m", formula="{a} * {b}")
    with pytest.raises(FormulaError, match="not antecedents"):
        kb.justify("e", 2, antecedents=["a"], source="model:m", formula="{b} - 1")
    assert "d" not in kb.keys(status=None)


# -- retraction and propagation ------------------------------------------------------------


def test_retraction_cascades(revenue_kb: BeliefBase) -> None:
    kb = revenue_kb
    kb.retract("revenue:Q2", reason="restated")
    assert kb.status("growth:Q3_vs_Q2") is Status.OUT
    assert kb.status("trend:Q3") is Status.OUT
    assert kb.status("risk:fx") is Status.IN
    assert kb.why_out("growth:Q3_vs_Q2") == "lost support: revenue:Q2"
    assert kb.why_out("revenue:Q2") == "retracted: restated"
    assert kb.confidence("trend:Q3") == 0.0


def test_readme_scenario_repairs_itself(revenue_kb: BeliefBase) -> None:
    kb = revenue_kb
    kb.retract("revenue:Q2", reason="restated in 10-K/A")
    kb.assert_("revenue:Q2", 4.1e9, source="tool:get_revenue")
    result = kb.propagate(include_kept=True)
    assert kinds(result) == [
        ("OUT", "revenue:Q2"),
        ("OUT", "growth:Q3_vs_Q2"),
        ("OUT", "trend:Q3"),
        ("IN", "revenue:Q2"),
        ("IN", "growth:Q3_vs_Q2"),
        ("IN", "trend:Q3"),
        ("KEPT", "risk:fx"),
    ]
    assert kb.value("growth:Q3_vs_Q2") == pytest.approx(9.75609756)
    assert kb.value("trend:Q3") == "strong"
    assert [b.key for b in result.rederived] == ["growth:Q3_vs_Q2", "trend:Q3"]
    assert "independent of revenue:Q2" in str(result.kept[0])
    assert result.pending == [] and result.conflicts == []
    # the old revisions stay in history, OUT
    assert kb.status("growth:Q3_vs_Q2@1") is Status.OUT
    assert kb.latest("growth:Q3_vs_Q2").revision == 2


def test_propagate_without_replacement_reports_pending(revenue_kb: BeliefBase) -> None:
    kb = revenue_kb
    kb.retract("revenue:Q2")
    result = kb.propagate()
    pending = {p.belief.key: p.reason for p in result.pending}
    assert pending == {"growth:Q3_vs_Q2": "waiting for: revenue:Q2", "trend:Q3": "waiting for: growth:Q3_vs_Q2"}
    assert "PEND" in str(result)


def test_early_cutoff_stops_cascade_when_value_unchanged(kb: BeliefBase) -> None:
    kb.assert_("revenue:Q2", 4.3e9, source="tool:a")
    kb.assert_("revenue:Q3", 4.5e9, source="tool:a")
    kb.derive("growth", growth, "revenue:Q2", "revenue:Q3")
    kb.derive("trend", trend, "growth")
    kb.changes()
    # A different, but still "modest", growth figure
    kb.retract("revenue:Q2")
    kb.assert_("revenue:Q2", 4.35e9, source="tool:a")
    result = kb.propagate()
    assert kb.latest("trend").revision == 1, "trend kept its revision: the cascade was cut off"
    assert kb.status("trend@1") is Status.IN
    assert ("OUT", "trend") not in kinds(result), "net change for trend is nothing"
    assert kb.latest("growth").revision == 2


def test_unchanged_rederivation_reuses_revision(kb: BeliefBase) -> None:
    kb.assert_("a", 2)
    kb.derive("b", rule(lambda v: v > 0, name="positive"), "a")
    kb.retract("a")
    kb.assert_("a", 3)
    kb.propagate()
    assert kb.latest("b").revision == 1
    assert kb.status("b") is Status.IN
    assert len(kb.justifications("b")) == 2


def test_restore_undoes_retraction(revenue_kb: BeliefBase) -> None:
    kb = revenue_kb
    kb.retract("revenue:Q2")
    kb.restore("revenue:Q2")
    assert kb.status("growth:Q3_vs_Q2") is Status.IN
    assert kb.changes() == []  # net: nothing changed
    with pytest.raises(NotBelievedError):
        kb.restore("revenue:Q3")


def test_retract_errors(kb: BeliefBase) -> None:
    kb.assert_("a", 1)
    kb.retract("a")
    with pytest.raises(NotBelievedError):
        kb.retract("a")
    with pytest.raises(UnknownBeliefError):
        kb.retract("zzz")
    with pytest.raises(UnknownBeliefError):
        kb.retract("a@9")


def test_retracting_a_ref_only_affects_that_revision(kb: BeliefBase) -> None:
    kb.assert_("x", 1, source="tool:a")
    kb.assert_("x", 2, source="tool:b")
    kb.retract("x@1")
    assert kb.value("x") == 2


# -- correct() ------------------------------------------------------------------------------


class TestCorrect:
    """``correct()``: retract the currently believed revision and assert the new value in one
    call, built on assert_()'s own supersede=True rollback."""

    def test_old_revision_out_with_reason_new_revision_in(self, kb: BeliefBase) -> None:
        q2 = kb.assert_("revenue:Q2", 4.3e9, source="tool:filings")
        corrected = kb.correct("revenue:Q2", 4.1e9, reason="restated in 10-K/A")

        assert kb.status(q2.ref) is Status.OUT
        assert kb.why_out(q2.ref) == "retracted: restated in 10-K/A"
        assert kb.status(corrected.ref) is Status.IN
        assert kb.value("revenue:Q2") == 4.1e9

    def test_the_custom_reason_also_reaches_the_change_log(self, kb: BeliefBase) -> None:
        """why_out() reads retract_reason live, but changes()/propagate() read a reason baked in
        by _relabel() during assert_(supersede=True) -- both must show the custom reason, not
        assert_'s own "superseded by <ref>" text."""
        kb.assert_("revenue:Q2", 4.3e9, source="tool:filings")
        kb.changes()
        kb.correct("revenue:Q2", 4.1e9, reason="restated in 10-K/A")

        (out_change,) = [c for c in kb.changes() if c.key == "revenue:Q2" and c.kind.value == "OUT"]
        assert out_change.reason == "retracted: restated in 10-K/A"

    def test_source_defaults_to_the_corrected_revisions_own_source(self, kb: BeliefBase) -> None:
        kb.assert_("revenue:Q2", 4.3e9, source="tool:filings")
        corrected = kb.correct("revenue:Q2", 4.1e9, reason="restated")
        assert str(corrected.source) == "tool:filings"

    def test_source_can_be_overridden(self, kb: BeliefBase) -> None:
        kb.assert_("revenue:Q2", 4.3e9, source="tool:filings")
        corrected = kb.correct("revenue:Q2", 4.1e9, source="human:analyst", reason="restated")
        assert str(corrected.source) == "human:analyst"

    def test_fault_source_records_a_wrong_outcome_in_the_ledger(self, kb: BeliefBase) -> None:
        kb.assert_("x", 1, source="tool:a")
        before = kb.reliability("tool:a")
        kb.correct("x", 2, source="tool:a", fault="source", reason="was wrong")
        assert kb.reliability("tool:a") < before

    def test_fault_none_does_not_touch_the_ledger(self, kb: BeliefBase) -> None:
        kb.assert_("x", 1, source="tool:a")
        before = kb.reliability("tool:a")
        kb.correct("x", 2, source="tool:a", reason="the world changed")
        assert kb.reliability("tool:a") == before

    def test_fault_source_on_a_derived_belief_is_rejected(self, kb: BeliefBase) -> None:
        kb.assert_("a", 1, source="tool:x")
        kb.derive("b", rule(lambda v: v * 2, name="double"), "a")
        with pytest.raises(ValueError, match="no source is at fault"):
            kb.correct("b", 99, source="tool:x", fault="source")

    def test_invalid_fault_is_rejected(self, kb: BeliefBase) -> None:
        kb.assert_("x", 1, source="tool:a")
        with pytest.raises(ValueError, match="fault must be"):
            kb.correct("x", 2, fault="everyone")

    def test_correcting_an_unknown_key_raises_unknown_belief_error(self, kb: BeliefBase) -> None:
        with pytest.raises(UnknownBeliefError):
            kb.correct("nope", 1)

    def test_a_failed_assertion_changes_nothing(self, kb: BeliefBase) -> None:
        kb.assert_("y", 1, source="tool:a")
        kb.changes()  # drain the initial assert's change, so only correct()'s effect is checked
        with pytest.raises(ValueError, match="reserved"):
            kb.correct("y", 2, source="rule:bad")
        assert kb.value("y") == 1
        assert kb.status("y") is Status.IN
        assert kb.changes() == []

    def test_accepts_a_belief_object(self, kb: BeliefBase) -> None:
        b = kb.assert_("z", 1, source="tool:a")
        corrected = kb.correct(b, 2, reason="fixed")
        assert corrected.value == 2

    def test_rejects_neither_string_nor_belief(self, kb: BeliefBase) -> None:
        kb.assert_("x", 1, source="tool:a")
        with pytest.raises(TypeError, match="expected a key or Belief"):
            kb.correct(123, 2)

    def test_forwards_extra_assert_kwargs(self, kb: BeliefBase) -> None:
        kb.assert_("x", 1, source="tool:a")
        corrected = kb.correct("x", 2, source="tool:a", claim="corrected value")
        assert corrected.claim == "corrected value"


def test_multiple_justifications_keep_belief_in(kb: BeliefBase) -> None:
    kb.assert_("a", 1)
    kb.assert_("b", 1)
    ident = rule(lambda v: v, name="ident")
    kb.derive("c", ident, "a")
    kb.derive("c", ident, "b")  # same value: a second, independent justification
    kb.retract("a")
    assert kb.status("c") is Status.IN
    kb.retract("b")
    assert kb.status("c") is Status.OUT


def test_support_cycles_are_not_self_supporting(kb: BeliefBase) -> None:
    kb.assert_("seed", 1)
    ident = rule(lambda v: v, name="ident")
    kb.derive("p", ident, "seed")
    kb.derive("q", ident, "p")
    kb.derive("p", ident, "q")  # p and q now also support each other
    assert kb.status("p") is Status.IN and kb.status("q") is Status.IN
    kb.retract("seed")
    assert kb.status("p") is Status.OUT, "a cycle alone is not well-founded support"
    assert kb.status("q") is Status.OUT


# -- non-monotonic justifications ----------------------------------------------------------


def test_unless_defeats_and_reinstates(kb: BeliefBase) -> None:
    kb.assert_("filing", "10-K")
    kb.derive("figures_final", rule(lambda f: True, name="final"), "filing", unless=["restatement"])
    assert kb.status("figures_final") is Status.IN
    kb.assert_("restatement", True, source="document:10-K/A")
    assert kb.status("figures_final") is Status.OUT
    assert kb.why_out("figures_final") == "defeated by: restatement"
    kb.retract("restatement")
    assert kb.status("figures_final") is Status.IN


def test_odd_loop_is_rejected_and_rolled_back(kb: BeliefBase) -> None:
    kb.assert_("seed", 1)
    ident = rule(lambda v: v, name="ident")
    kb.derive("a", ident, "seed", unless=["b"])
    with pytest.raises(CircularDefeatError):
        kb.derive("b", ident, "a")
    assert "b" not in kb.keys(status=None)
    assert kb.status("a") is Status.IN
    assert kb.changes()[-1].key == "a", "no spurious changes from the failed attempt"


def test_self_defeat_is_rejected(kb: BeliefBase) -> None:
    kb.assert_("seed", 1)
    with pytest.raises(CircularDefeatError):
        kb.derive("a", rule(lambda v: v, name="ident"), "seed", unless=["a"])


# -- constraints and conflicts -------------------------------------------------------------


def test_constraint_conflict_and_resolution(kb: BeliefBase) -> None:
    kb.add_constraint("margin<=100", ["margin"], lambda m: m <= 100, description="margin above 100%")
    kb.assert_("margin", 130, source="tool:calc")
    (conflict,) = kb.conflicts()
    assert conflict.kind.value == "constraint" and conflict.description == "margin above 100%"
    resolution = kb.resolve(conflict, retract="margin")
    assert resolution.retract == ("margin@1",)
    assert kb.conflicts() == []


def test_constraint_that_raises_is_a_conflict(kb: BeliefBase) -> None:
    kb.add_constraint("positive", ["x"], lambda x: x > 0)
    kb.assert_("x", "text")
    (conflict,) = kb.conflicts()
    assert "raised" in conflict.description


def test_add_constraint_validates_arguments(kb: BeliefBase) -> None:
    with pytest.raises(ValueError):
        kb.add_constraint("c")


def test_resolve_value_conflict_keep(kb: BeliefBase) -> None:
    kb.assert_("x", 1, source="tool:a")
    kb.assert_("x", 2, source="human:alice")
    (conflict,) = kb.conflicts()
    kb.resolve(conflict, keep="x@2")
    assert kb.value("x") == 2
    with pytest.raises(ValueError):
        kb.resolve(conflict)
    with pytest.raises(ValueError, match="not part of conflict"):
        kb.assert_("y", 1)
        kb.resolve(conflict, keep="y")


def test_conflict_explain_contains_both_chains(kb: BeliefBase) -> None:
    kb.assert_("x", 1, source="tool:a")
    kb.assert_("x", 2, source="tool:b")
    text = kb.conflicts()[0].explain()
    assert "tool:a" in text and "tool:b" in text


# -- documents -------------------------------------------------------------------------------


def test_cite_checks_quote_and_value(kb: BeliefBase) -> None:
    kb.add_document("10-K", "Total revenue for the quarter was $4.3 billion, up from last year.")
    b = kb.cite("revenue:Q2", 4.3e9, document="10-K", quote="Total revenue for the quarter was $4.3 billion")
    assert b.source == Source.document("10-K", quote="Total revenue for the quarter was $4.3 billion")
    with pytest.raises(CitationError, match="not found"):
        kb.cite("x", 1, document="10-K", quote="revenue was $9 billion")
    with pytest.raises(CitationError, match="not stated"):
        kb.cite("x", 5e9, document="10-K", quote="Total revenue for the quarter was $4.3 billion")
    with pytest.raises(CitationError, match="unknown document"):
        kb.cite("x", 1, document="nope", quote="x")
    with pytest.raises(ValueError):
        kb.add_document("", "text")


# -- queries and views -------------------------------------------------------------------------


def test_explain_and_transcript(revenue_kb: BeliefBase) -> None:
    kb = revenue_kb
    text = kb.explain("growth:Q3_vs_Q2")
    assert "rule:growth" in text and "used by:" in text and "trend:Q3@1" in text
    kb.retract("revenue:Q2", reason="restated")
    assert "why OUT:    lost support: revenue:Q2" in kb.explain("growth:Q3_vs_Q2")
    transcript = kb.transcript()
    assert "retract  revenue:Q2@1  restated" in transcript
    assert kb.history[0].action == "assert"


def test_listing(revenue_kb: BeliefBase) -> None:
    kb = revenue_kb
    kb.retract("fx:exposure")
    assert "fx:exposure" in kb.keys(Status.OUT)
    assert "fx:exposure" not in kb.keys()  # noqa: SIM118 (BeliefBase.keys takes a status filter)
    assert len(kb.keys(None)) == 6
    assert {b.key for b in kb} == {"revenue:Q2", "revenue:Q3", "growth:Q3_vs_Q2", "trend:Q3"}
    assert len(kb) == 4
    assert len(kb.beliefs(None)) == 6
    assert "BeliefBase" in repr(kb)


def test_relabel_only_touches_the_affected_region(revenue_kb: BeliefBase) -> None:
    kb = revenue_kb
    calls: list[int] = []
    original = kb._label_components

    def spy(region: dict[str, None], now: object) -> list[str]:
        calls.append(len(region))
        return original(region, now)  # type: ignore[arg-type]

    kb._label_components = spy  # type: ignore[method-assign]
    kb.retract("fx:exposure")
    assert calls == [2], "only fx:exposure and risk:fx were relabeled"


def test_changes_buffer_is_net(kb: BeliefBase) -> None:
    kb.assert_("a", 1)
    assert [c.kind for c in kb.changes()] == [ChangeKind.IN]
    assert kb.changes() == []


# -- robustness -------------------------------------------------------------------------------


def test_naive_valid_until_is_rejected_before_anything_changes(kb: BeliefBase) -> None:
    from datetime import datetime

    with pytest.raises(ValueError, match="timezone-aware"):
        kb.assert_("b", 2, source="tool:x", valid_until=datetime(2030, 1, 1))
    assert kb.keys(status=None) == []


def test_a_failed_assertion_leaves_the_base_unchanged(kb: BeliefBase, monkeypatch: pytest.MonkeyPatch) -> None:
    kb.assert_("a", 1, source="tool:x")
    before = kb.to_dict()
    relabel = kb._relabel

    def broken(seeds: list[str]) -> None:
        monkeypatch.setattr(kb, "_relabel", relabel)  # fail once, then let the rollback relabel
        raise RuntimeError("boom")

    monkeypatch.setattr(kb, "_relabel", broken)
    with pytest.raises(RuntimeError):
        kb.assert_("a", 2, source="tool:y", supersede=True)
    assert kb.status("a@1") is Status.IN
    assert [b.ref for b in kb.revisions("a")] == ["a@1"]
    assert BeliefBase.from_dict(kb.to_dict()).value("a") == 1
    assert kb.to_dict()["beliefs"] == before["beliefs"]


def test_a_lone_string_is_one_key_not_its_characters(kb: BeliefBase) -> None:
    kb.assert_("price", 10, source="tool:x")
    kb.derive("discounted", lambda p: p * 0.9, "price", unless="override")
    support = kb.support("discounted")
    assert support is not None and support.unless == ("override",)
    kb.justify("double", 20, antecedents="price", source="model:m", inputs="price")
    support = kb.support("double")
    assert support is not None and support.antecedents == ("price@1",) and support.inputs == ("price",)
    constraint = kb.add_constraint("positive", "price", lambda p: p > 0)
    assert constraint.keys == ("price",)


def test_supersede_resolves_the_conflict_when_the_value_already_exists(kb: BeliefBase) -> None:
    kb.assert_("k", 1, source="tool:a")
    kb.assert_("k", 2, source="tool:b")
    assert [c.id for c in kb.conflicts()] == ["value:k"]
    belief = kb.assert_("k", 2, source="tool:c", supersede=True)
    assert belief.ref == "k@2"
    assert kb.conflicts() == []
    assert kb.status("k@1") is Status.OUT and kb.value("k") == 2


def test_retracting_a_retracted_ref_does_not_blame_its_source_twice(kb: BeliefBase) -> None:
    kb.assert_("a", 1, source="tool:x")
    kb.retract("a@1", fault="source")
    assert kb.retract("a@1", fault="source") == []
    assert kb.ledger.record_of("tool:x").wrong == 1


def test_resolve_with_an_ambiguous_key_asks_for_a_ref(kb: BeliefBase) -> None:
    kb.assert_("k", 1, source="tool:a")
    kb.assert_("k", 2, source="tool:b")
    (conflict,) = kb.conflicts()
    with pytest.raises(ValueError, match="several sides"):
        kb.resolve(conflict, keep="k")
    kb.resolve(conflict, keep="k@2")
    assert kb.value("k") == 2


def test_a_resolver_naming_refs_outside_the_conflict_is_an_error(kb: BeliefBase) -> None:
    from corollary import Resolution

    kb.assert_("k", 1, source="tool:a")
    kb.assert_("k", 2, source="tool:b")
    with pytest.raises(ValueError, match="nope@1"):
        kb.resolve_conflicts(lambda c, _kb: Resolution(c.id, ("nope@1",), "bad"))


def test_confidence_values_are_validated_everywhere() -> None:
    from corollary import Rule
    from corollary.kernel import Derived

    with pytest.raises(ValueError, match="between 0 and 1"):
        Rule("r", lambda: 1, confidence=5.0)
    with pytest.raises(ValueError, match="needs a name"):
        Rule("", lambda: 1)
    with pytest.raises(ValueError, match="between 0 and 1"):
        Derived(1, confidence=1.5)


def test_propagation_settled(kb: BeliefBase) -> None:
    kb.assert_("k", 1, source="tool:a")
    kb.assert_("k", 2, source="tool:b")
    result = kb.propagate()
    assert not result.settled and result.conflicts
    kb.resolve(result.conflicts[0], keep="k@2")
    assert kb.propagate().settled


def test_rules_can_be_replaced_and_same_named_functions_coexist(kb: BeliefBase) -> None:
    kb.register_rule(growth)
    kb.register_rule(rule(lambda a, b: 0, name="growth"), replace=True)
    assert kb.rules["growth"].fn(1, 2) == 0

    def make(factor: float):  # type: ignore[no-untyped-def]
        def scale(x: float) -> float:
            return x * factor

        return scale

    kb.assert_("x", 10)
    assert kb.derive("double", make(2), "x").value == 20
    assert kb.derive("triple", make(3), "x").value == 30
    assert {"scale", "scale:2"} <= set(kb.rules)


def test_auto_named_rules_never_reuse_a_name_a_snapshot_still_uses(kb: BeliefBase) -> None:
    from corollary import Rule

    def make(factor: float):  # type: ignore[no-untyped-def]
        def scale(x: float) -> float:
            return x * factor

        return scale

    kb.assert_("x", 1, source="tool:a")
    kb.derive("a", make(1), "x")
    kb.derive("b", make(2), "x")  # rule "scale:2"
    with pytest.warns(UserWarning, match="scale:2"):
        loaded = BeliefBase.from_dict(kb.to_dict(), rules=[Rule("scale", make(1))])
    loaded.derive("c", make(10), "x")
    assert loaded.support("c").rule not in {"scale", "scale:2"}  # type: ignore[union-attr]
    loaded.register_rule(Rule("scale:4", make(4)))
    loaded.derive("d", make(5), "x")  # skips the explicitly registered name too
    assert loaded.value("d") == 5


def test_rollback_restores_justification_order_and_ledger(kb: BeliefBase, monkeypatch: pytest.MonkeyPatch) -> None:
    kb.assert_("x", 1, source="tool:a")
    kb.assert_("x", 1, source="human:bob")
    before = [j.id for j in kb.justifications("x")]
    support = kb.support("x")
    ledger_version = kb.ledger.version
    relabel = kb._relabel

    def broken(seeds: list[str]) -> None:
        monkeypatch.setattr(kb, "_relabel", relabel)
        raise RuntimeError("boom")

    monkeypatch.setattr(kb, "_relabel", broken)
    with pytest.raises(RuntimeError):
        kb.assert_("x", 1, source="tool:a")  # a re-read: replaces tool:a's justification
    assert [j.id for j in kb.justifications("x")] == before
    assert kb.support("x") == support
    monkeypatch.setattr(kb, "_relabel", broken)
    with pytest.raises(RuntimeError):
        kb.assert_("x", 1, source="tool:c")  # a confirmation that fails is not credited
    assert kb.ledger.version == ledger_version


def test_a_re_read_is_logged_as_a_renewal(kb: BeliefBase) -> None:
    kb.assert_("x", 1, source="tool:a")
    kb.assert_("x", 1, source="tool:a")
    assert [e.action for e in kb.history][-1] == "renew"
