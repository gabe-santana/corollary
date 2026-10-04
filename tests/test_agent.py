from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from corollary import (
    Agent,
    BeliefBase,
    CorollaryError,
    Dependencies,
    JustificationKind,
    ModelError,
    PreferSource,
    ScriptedModel,
    Status,
    tool,
)

from .conftest import Clock

GROWTH = "({revenue:Q3} - {revenue:Q2}) / {revenue:Q2} * 100"
REVENUE = {"Q2": 4.3e9, "Q3": 4.5e9}


@tool(trust="high")
def get_revenue(quarter: str) -> float:
    """Quarterly revenue in USD."""
    return REVENUE[quarter]


def call(quarter: str) -> dict[str, Any]:
    return {"type": "call_tool", "tool": "get_revenue", "args": {"quarter": quarter}, "key": f"revenue:{quarter}"}


def claim(key: str, value: Any = None, follows: list[str] | None = None, **extra: Any) -> dict[str, Any]:
    action = {"type": "claim", "key": key, "follows_from": follows or [], **extra}
    if value is not None:
        action["value"] = value
    return action


def answer(text: str, follows: list[str]) -> dict[str, Any]:
    return {"type": "answer", "text": text, "follows_from": follows}


def actions(*items: dict[str, Any]) -> dict[str, Any]:
    return {"actions": list(items)}


FETCH = actions(call("Q2"), call("Q3"))
ANALYZE = actions(
    claim("growth:Q3_vs_Q2", None, ["revenue:Q2", "revenue:Q3"], formula=GROWTH, claim="Q3 revenue grew 4.65% over Q2"),
    claim("trend", "modest", ["growth:Q3_vs_Q2"], claim="Growth is modest"),
    answer("Q3 revenue grew 4.65% over Q2: modest growth.", ["growth:Q3_vs_Q2", "trend"]),
)


@pytest.fixture(autouse=True)
def reset_revenue() -> None:
    REVENUE.update({"Q2": 4.3e9, "Q3": 4.5e9})


def make_agent(*responses: Any, **kwargs: Any) -> tuple[Agent, ScriptedModel]:
    model = ScriptedModel(responses)
    return Agent(model, tools=[get_revenue], **kwargs), model


def test_run_produces_answer_with_proof() -> None:
    agent, model = make_agent(FETCH, ANALYZE, dependencies="declared")
    report = agent.run("Compare Q2 and Q3 revenue.")
    assert report.completed and not report.stale
    assert report.answer == "Q3 revenue grew 4.65% over Q2: modest growth."
    assert str(report) == report.answer
    assert report.key == "answer"
    assert agent.kb.value("growth:Q3_vs_Q2") == pytest.approx(4.651162790697675)
    assert [s.belief.key for s in report.proof.premises] == ["revenue:Q2", "revenue:Q3"]
    assert report.verify().ok
    assert report.verify(raise_on_error=True).ok
    assert len(report.steps) == 2 and report.rejections == []
    assert [c.key for c in report.changes][:2] == ["revenue:Q2", "revenue:Q3"]
    assert model.remaining == 0
    assert agent.beliefs is agent.kb


def test_report_repr_summarizes_run_without_prompts() -> None:
    agent, _ = make_agent(FETCH, ANALYZE, dependencies="declared")
    report = agent.run("Compare Q2 and Q3 revenue.")

    rendered = repr(report)
    assert "task='Compare Q2 and Q3 revenue.'" in rendered
    assert "completed=True" in rendered
    assert "steps=2" in rendered
    assert report.steps[0].prompt not in rendered
    assert str(report) == report.answer

    report.task = "x" * 61
    assert f"task={('x' * 57 + '...')!r}" in repr(report)


def test_tool_results_are_premises_recorded_by_the_runtime() -> None:
    agent, _ = make_agent(FETCH, ANALYZE)
    agent.run("t")
    support = agent.kb.support("revenue:Q2")
    assert support is not None and support.kind is JustificationKind.PREMISE
    assert str(support.source) == "tool:get_revenue(quarter='Q2')"
    assert agent.kb.confidence("revenue:Q2") == pytest.approx(0.99)


def test_projection_hides_nothing_but_shows_only_in_beliefs() -> None:
    agent, model = make_agent(FETCH, ANALYZE)
    agent.run("t")
    second_prompt = model.calls[1].prompt
    assert "- revenue:Q2 = 4300000000.0" in second_prompt
    assert "get_revenue(quarter: str) -> float" in second_prompt


def test_conservative_dependencies_cover_the_whole_context() -> None:
    agent, _ = make_agent(FETCH, ANALYZE)
    agent.run("t")
    support = agent.kb.support("trend")
    assert support is not None
    # trend only declared growth, but it was generated in a context that showed the revenues too
    assert set(support.inputs) == {"revenue:Q2", "revenue:Q3", "growth:Q3_vs_Q2"}


def test_declared_dependencies_follow_follows_from() -> None:
    agent, _ = make_agent(FETCH, ANALYZE, dependencies=Dependencies.DECLARED)
    agent.run("t")
    support = agent.kb.support("trend")
    assert support is not None and support.inputs == ("growth:Q3_vs_Q2",)


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (claim("x", 1, ["revenue:Q9"]), "unknown key 'revenue:Q9'"),
        (claim("x", 99, ["revenue:Q2", "revenue:Q3"], formula=GROWTH), "but the claim states 99"),
        (claim("revenue:Q2", 1.0, []), "already holds 4,300,000,000"),
        (claim("bad key", 1, []), "invalid belief key"),
        (claim("answer", 1, []), "reserved for the answer"),
        ({**call("Q2"), "key": "answer"}, "reserved for the answer"),
        ({"type": "cite", "document": "none", "quote": "q", "key": "answer", "value": 1}, "reserved for the answer"),
        ({"type": "call_tool", "tool": "launch_missiles", "args": {}}, "unknown tool 'launch_missiles'"),
        ({"type": "call_tool", "tool": "get_revenue", "args": {"q": "Q2"}}, "invalid arguments"),
        ({"type": "call_tool", "tool": "get_revenue", "args": {"quarter": "Q7"}}, "raised KeyError"),
        ({"type": "cite", "document": "none", "quote": "q", "key": "k", "value": 1}, "unknown document"),
        ({"type": "teleport"}, "unknown action type"),
    ],
)
def test_invalid_actions_are_rejected_and_fed_back(bad: dict[str, Any], message: str) -> None:
    agent, model = make_agent(FETCH, actions(bad), ANALYZE)
    report = agent.run("t")
    assert report.completed
    assert any(message in r for r in report.rejections), report.rejections
    assert "# Runtime feedback" in model.calls[2].prompt
    assert message in model.calls[2].prompt


def test_model_error_ends_the_run_but_keeps_the_report() -> None:
    agent, _ = make_agent(FETCH)  # the scripted model has nothing to say on step 2
    report = agent.run("t")
    assert not report.completed and report.answer is None
    assert isinstance(report.error, ModelError) and "no responses left" in str(report.error)
    assert len(report.steps) == 2 and report.steps[0].accepted
    assert report.rejections == ["model error: ScriptedModel has no responses left"]
    assert agent.kb.value("revenue:Q2") == 4.3e9


def test_async_tools_and_string_arguments() -> None:
    @tool
    async def revenue_in_billions(quarter: str, scale: float) -> float:
        return REVENUE[quarter] / scale

    call_it = {
        "type": "call_tool",
        "tool": "revenue_in_billions",
        "args": {"quarter": "Q2", "scale": "1e9"},
        "key": "rev:Q2",
    }
    agent = Agent(ScriptedModel([actions(call_it), actions(answer("4.3B", ["rev:Q2"]))]))
    agent.tools["revenue_in_billions"] = revenue_in_billions
    report = agent.run("t")
    assert report.completed, report.rejections
    assert agent.kb.value("rev:Q2") == pytest.approx(4.3)


def test_claim_cannot_use_a_tool_result_from_the_same_turn() -> None:
    agent, _ = make_agent(
        actions(call("Q2"), call("Q3"), claim("g", 1.0, ["revenue:Q2"])),
        ANALYZE,
    )
    report = agent.run("t")
    assert any("tool called in this same response" in r for r in report.rejections)
    assert "g" not in agent.kb.keys(status=None)


def test_claim_cannot_use_beliefs_hidden_from_the_context() -> None:
    agent, _ = make_agent(FETCH, ANALYZE)
    agent.kb.assert_("secret", 1)
    agent.kb.retract("secret")
    agent.model.responses.insert(1, actions(claim("leak", 1, ["secret"])))  # type: ignore[attr-defined]
    report = agent.run("t")
    assert any("not in this turn's context" in r for r in report.rejections)


def test_claims_may_build_on_earlier_claims_in_the_same_turn() -> None:
    agent, _ = make_agent(FETCH, ANALYZE)
    report = agent.run("t")
    assert report.rejections == []
    assert agent.kb.status("trend") is Status.IN


def test_formula_alone_computes_the_value() -> None:
    agent, _ = make_agent(FETCH, ANALYZE)
    agent.run("t")
    assert agent.kb.support("growth:Q3_vs_Q2").formula == GROWTH  # type: ignore[union-attr]
    assert agent.kb.confidence("growth:Q3_vs_Q2") == pytest.approx(0.99), "formula steps carry rule trust"
    assert agent.kb.confidence("trend") == pytest.approx(0.99 * 0.9)


def test_unparseable_and_empty_responses_get_feedback() -> None:
    agent, model = make_agent("not json at all", actions(), FETCH, ANALYZE)
    report = agent.run("t")
    assert report.completed
    assert "not valid JSON" in model.calls[1].prompt
    assert "no actions" in model.calls[2].prompt


def test_step_budget() -> None:
    agent, _ = make_agent(FETCH, FETCH, FETCH)
    report = agent.run("t", max_steps=2)
    assert not report.completed
    assert report.answer is None and report.belief is None
    with pytest.raises(CorollaryError):
        _ = report.proof
    assert str(report) == "(no answer)"


def test_repeated_tool_call_supersedes_its_own_result() -> None:
    agent, _ = make_agent(actions(call("Q2")), actions(call("Q2")), actions(answer("ok", ["revenue:Q2"])))
    REVENUE["Q2"] = 4.3e9
    original = get_revenue.fn
    values = iter([4.3e9, 4.2e9])
    get_revenue.fn = lambda quarter: next(values)
    try:
        agent.run("t")
    finally:
        get_revenue.fn = original
    assert agent.kb.value("revenue:Q2") == 4.2e9
    assert agent.kb.conflicts() == []


def test_citations_through_the_agent() -> None:
    doc = "ACME 10-K: Total revenue for Q2 was $4.3 billion."
    model = ScriptedModel(
        [
            actions(
                {
                    "type": "cite",
                    "document": "10-K",
                    "quote": "Total revenue for Q2 was $4.3 billion",
                    "key": "revenue:Q2",
                    "value": 4.3e9,
                },
                {
                    "type": "cite",
                    "document": "10-K",
                    "quote": "Total revenue for Q2 was $9 billion",
                    "key": "fake",
                    "value": 9e9,
                },
            ),
            actions(answer("Q2 revenue was $4.3 billion.", ["revenue:Q2"])),
        ]
    )
    agent = Agent(model, documents={"10-K": doc})
    report = agent.run("What was Q2 revenue?")
    assert any("quote not found" in r for r in report.rejections)
    assert report.verify().ok
    assert "## 10-K" in model.calls[0].prompt


def test_answer_keys_are_unique_per_run() -> None:
    agent, _ = make_agent(FETCH, ANALYZE, actions(answer("again", ["trend"])))
    first = agent.run("t")
    second = agent.run("t2")
    assert (first.key, second.key) == ("answer", "answer:2")


def test_declared_answer_needs_support() -> None:
    agent, _ = make_agent(FETCH, actions(answer("vibes", [])), ANALYZE, dependencies="declared")
    report = agent.run("t")
    assert any("must list the beliefs" in r for r in report.rejections)


def test_resolver_runs_before_each_step() -> None:
    model = ScriptedModel([actions(answer("ok", ["x"]))])
    kb = BeliefBase()
    kb.assert_("x", 1, source="human:alice")
    kb.assert_("x", 2, source="tool:scraper")
    agent = Agent(model, kb, resolver=PreferSource(["human", "tool"]))
    agent.run("t")
    assert kb.value("x") == 1
    assert "# Conflicts" not in model.calls[0].prompt


# -- repair ------------------------------------------------------------------------------------


def correct_q2(agent: Agent) -> None:
    agent.kb.retract("revenue:Q2", reason="restated in 10-K/A")
    agent.kb.assert_("revenue:Q2", 4.1e9, source="tool:get_revenue")


def test_repair_rederives_model_claims_and_answer() -> None:
    agent, model = make_agent(FETCH, ANALYZE)
    report = agent.run("Compare Q2 and Q3 revenue.")
    correct_q2(agent)
    assert report.stale and report.answer is None
    model.add(
        actions(claim("growth:Q3_vs_Q2", None, ["revenue:Q2", "revenue:Q3"], formula=GROWTH, claim="grew 9.76%")),
        actions(claim("trend", "strong", ["growth:Q3_vs_Q2"], claim="Growth is strong")),
        actions(claim("answer", "Q3 revenue grew 9.76% over Q2: strong growth.", ["growth:Q3_vs_Q2", "trend"])),
    )
    result = agent.repair(include_kept=True)
    assert result[0].key == "revenue:Q2" and result[0].kind.value == "OUT"
    out = {c.key for c in result.retracted}
    back = [c.key for c in result.added]
    assert out == {"revenue:Q2", "growth:Q3_vs_Q2", "trend", "answer"}
    assert back == ["revenue:Q2", "growth:Q3_vs_Q2", "trend", "answer"], "re-derived in dependency order"
    assert report.answer == "Q3 revenue grew 9.76% over Q2: strong growth."
    assert not report.stale
    assert report.verify().ok
    # The re-derivation prompt showed only the current inputs, never the retracted value.
    rederive_prompt = model.calls[2].prompt
    assert "Re-derive the belief `growth:Q3_vs_Q2`" in rederive_prompt
    assert "4100000000.0" in rederive_prompt and "4300000000.0" not in rederive_prompt.split("# Beliefs")[1]


def test_repair_rejects_invalid_rederivations_and_retries() -> None:
    agent, model = make_agent(FETCH, ANALYZE, repair_attempts=2)
    agent.run("t")
    correct_q2(agent)
    model.add(
        actions(claim("growth:Q3_vs_Q2", 1.0, ["revenue:Q2", "revenue:Q3"], formula=GROWTH)),  # wrong value
        actions(claim("growth:Q3_vs_Q2", None, ["revenue:Q2", "revenue:Q3"], formula=GROWTH)),
        actions(claim("trend", "strong", ["unrelated"])),  # outside its inputs
        actions(),  # gives up
    )
    result = agent.repair()
    assert agent.kb.value("growth:Q3_vs_Q2") == pytest.approx(9.75609756)
    assert "evaluates to" in model.calls[3].prompt
    pending = {p.belief.key for p in result.pending}
    assert pending == {"trend", "answer"}


def test_repair_retries_after_a_malformed_rederivation() -> None:
    agent, model = make_agent(FETCH, ANALYZE, repair_attempts=2)
    agent.run("t")
    correct_q2(agent)
    model.add(
        actions({"type": "claim", "formula": GROWTH}),  # no key: fails to parse
        actions(claim("growth:Q3_vs_Q2", None, ["revenue:Q2", "revenue:Q3"], formula=GROWTH)),
    )
    agent.repair()
    assert agent.kb.value("growth:Q3_vs_Q2") == pytest.approx(9.75609756)
    assert "claim requires 'key'" in model.calls[3].prompt


def test_repair_with_rules_needs_no_model() -> None:
    agent, model = make_agent(FETCH, actions(answer("ok", ["revenue:Q2"])))
    agent.run("t")
    agent.kb.derive("double", lambda r: r * 2, "revenue:Q2")
    agent.kb.retract("revenue:Q2")
    agent.kb.assert_("revenue:Q2", 1.0, source="tool:get_revenue")
    model.add(actions(claim("answer", "ok2", ["revenue:Q2"])))
    agent.repair()
    assert agent.kb.value("double") == 2.0


# -- reverify and narrow ---------------------------------------------------------------------


def test_reverify_refreshes_expired_tool_results(clock: Clock) -> None:
    calls = []

    @tool(ttl=timedelta(minutes=1))
    def price(symbol: str) -> float:
        calls.append(symbol)
        return 101.0

    model = ScriptedModel(
        [
            actions({"type": "call_tool", "tool": "price", "args": {"symbol": "ACME"}, "key": "price:ACME"}),
            actions(answer("ACME trades at 101.", ["price:ACME"])),
        ]
    )
    agent = Agent(model, BeliefBase(clock=clock), tools=[price])
    report = agent.run("Price of ACME?")
    clock.advance(minutes=2)
    agent.kb.refresh()
    assert report.stale
    agent.reverify()
    assert calls == ["ACME", "ACME"]
    assert not report.stale, "same value: evidence renewed in place, answer restored without the model"
    assert model.remaining == 0


def test_narrow_prunes_unneeded_dependencies() -> None:
    agent, model = make_agent(FETCH, ANALYZE)
    agent.kb.assert_("weather", "sunny", source="tool:weather")
    agent.run("t")
    assert "weather" in agent.kb.support("trend").inputs  # type: ignore[union-attr]

    def respond(prompt: str) -> dict[str, Any]:
        """A model that derives each belief only when its real inputs are visible."""
        visible = prompt.split("# Beliefs")[1]
        if "Derive the belief `growth:Q3_vs_Q2`" in prompt:
            if "revenue:Q2 =" in visible and "revenue:Q3 =" in visible:
                return actions(claim("growth:Q3_vs_Q2", None, ["revenue:Q2", "revenue:Q3"], formula=GROWTH))
        elif "growth:Q3_vs_Q2 =" in visible:
            return actions(claim("trend", "modest", ["growth:Q3_vs_Q2"]))
        return actions()

    model.add(*[respond] * 10)
    growth = agent.narrow("growth:Q3_vs_Q2")
    assert growth.kept == ("revenue:Q2", "revenue:Q3") and growth.pruned == ("weather",)
    trend = agent.narrow("trend")
    assert trend.kept == ("growth:Q3_vs_Q2",)
    assert set(trend.pruned) == {"revenue:Q2", "revenue:Q3", "weather"}
    assert trend.calls == 4
    agent.kb.retract("weather")
    assert agent.kb.status("growth:Q3_vs_Q2") is Status.IN
    assert agent.kb.status("trend") is Status.IN, "pruned dependency no longer cascades"
    assert agent.kb.status("answer") is Status.OUT, "the un-narrowed answer still over-retracts, safely"


def test_narrow_requires_model_support(revenue_kb: BeliefBase) -> None:
    agent = Agent(ScriptedModel(), revenue_kb)
    with pytest.raises(CorollaryError):
        agent.narrow("growth:Q3_vs_Q2")


def test_model_from_string_and_plain_function_tools() -> None:
    def add(a: int, b: int) -> int:
        return a + b

    agent = Agent("anthropic:claude-opus-5-5", tools=[add])
    assert agent.model.name == "claude-opus-5-5"
    assert "add" in agent.tools


def test_repair_follows_dependencies_even_with_identical_timestamps(clock: Clock) -> None:
    """Regression: on Windows, coarse clocks gave growth, trend and answer the same created_at,
    and re-derivation fell back to alphabetical order (answer first)."""
    model = ScriptedModel([FETCH, ANALYZE])
    agent = Agent(model, BeliefBase(clock=clock), tools=[get_revenue])  # the clock never advances
    report = agent.run("t")
    correct_q2(agent)
    model.add(
        actions(claim("growth:Q3_vs_Q2", None, ["revenue:Q2", "revenue:Q3"], formula=GROWTH)),
        actions(claim("trend", "strong", ["growth:Q3_vs_Q2"])),
        actions(claim("answer", "Q3 grew 9.76%: strong.", ["growth:Q3_vs_Q2", "trend"])),
    )
    result = agent.repair()
    assert [c.key for c in result.added] == ["revenue:Q2", "growth:Q3_vs_Q2", "trend", "answer"]
    assert report.answer == "Q3 grew 9.76%: strong."
    assert "Re-derive the belief `growth:Q3_vs_Q2`" in model.calls[2].prompt


# -- confidence: learning from checks, self-consistency, decay ----------------------------------------


def test_verified_formulas_and_citations_teach_the_ledger() -> None:
    model = ScriptedModel(
        [
            FETCH,
            actions(
                claim("g1", 4.651162790697675, ["revenue:Q2", "revenue:Q3"], formula=GROWTH),
                claim("g2", 99.0, ["revenue:Q2", "revenue:Q3"], formula=GROWTH),
                answer("done", ["g1"]),
            ),
        ]
    )
    agent = Agent(model, tools=[get_revenue])
    agent.run("t")
    record = agent.kb.ledger.record_of("model:scripted")
    assert (record.correct, record.wrong) == (1, 1)


def test_learning_from_checks_can_be_disabled() -> None:
    model = ScriptedModel([FETCH, actions(claim("g", 1.0, ["revenue:Q2", "revenue:Q3"], formula=GROWTH)), ANALYZE])
    agent = Agent(model, tools=[get_revenue], learn_from_checks=False)
    agent.run("t")
    assert agent.kb.ledger.sources() == []


def test_citation_outcomes() -> None:
    doc = "Total revenue for Q2 was $4.3 billion."
    model = ScriptedModel(
        [
            actions(
                {
                    "type": "cite",
                    "document": "10-K",
                    "quote": "Total revenue for Q2 was $4.3 billion",
                    "key": "revenue:Q2",
                    "value": 4.3e9,
                },
                {
                    "type": "cite",
                    "document": "10-K",
                    "quote": "Total revenue for Q2 was $5 billion",
                    "key": "fake",
                    "value": 5e9,
                },
                {"type": "cite", "document": "missing", "quote": "x", "key": "k", "value": 1},
            ),
            actions(answer("Q2 revenue was $4.3 billion.", ["revenue:Q2"])),
        ]
    )
    agent = Agent(model, documents={"10-K": doc})
    agent.run("t")
    record = agent.kb.ledger.record_of("model:scripted")
    assert (record.correct, record.wrong) == (1, 1), "an unknown document is a format error, not a wrong fact"


def test_self_consistency_scores_unverified_claims() -> None:
    def check(answer_value: str):  # type: ignore[no-untyped-def]
        def respond(prompt: str) -> dict[str, Any]:
            assert "Derive the belief `trend`" in prompt
            assert "modest" not in prompt.split("# Beliefs")[0], "samples never see the original answer"
            return actions(claim("trend", answer_value, ["growth:Q3_vs_Q2"]))

        return respond

    model = ScriptedModel([FETCH, ANALYZE, check("modest"), check("strong")])
    agent = Agent(model, tools=[get_revenue], dependencies="declared", self_consistency=3)
    agent.run("t")
    # 2 of 3 samples agree: certainty (2 + 1) / (3 + 1)
    assert agent.kb.support("trend").confidence == pytest.approx(0.75)  # type: ignore[union-attr]
    assert agent.kb.confidence("trend") == pytest.approx(0.9 * 0.75 * 0.99)
    assert model.remaining == 0, "the formula claim and the answer were not sampled"


def test_self_consistency_validation() -> None:
    with pytest.raises(ValueError):
        Agent(ScriptedModel(), self_consistency=0)


def test_reverify_refreshes_faded_tool_results(clock: Clock) -> None:
    calls = []

    @tool(half_life=timedelta(minutes=1), origin="exchange")
    def price(symbol: str) -> float:
        calls.append(symbol)
        return 101.0

    model = ScriptedModel(
        [
            actions({"type": "call_tool", "tool": "price", "args": {"symbol": "ACME"}, "key": "price:ACME"}),
            actions(answer("ACME trades at 101.", ["price:ACME"])),
        ]
    )
    from corollary import TrustPolicy

    agent = Agent(model, BeliefBase(clock=clock, trust=TrustPolicy(min_confidence=0.5)), tools=[price])
    report = agent.run("Price of ACME?")
    assert agent.kb["price:ACME"].source.origin == "exchange"
    clock.advance(seconds=70)
    assert agent.kb.confidence("price:ACME") < 0.5
    assert not report.stale, "faded, not expired: still believed"
    agent.reverify()
    assert calls == ["ACME", "ACME"]
    assert agent.kb.confidence("price:ACME") == pytest.approx(0.99)
    assert model.remaining == 0


def test_step_budget_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_steps"):
        Agent(ScriptedModel(), max_steps=0)
    agent, _ = make_agent(FETCH)
    with pytest.raises(ValueError, match="max_steps"):
        agent.run("t", max_steps=0)


def test_instructions_reach_every_prompt() -> None:
    agent, model = make_agent(FETCH, ANALYZE, instructions="Answer in Portuguese.")
    agent.run("t", instructions="Be brief.")
    assert "# Instructions\nAnswer in Portuguese.\n\nBe brief." in model.calls[0].prompt
    correct_q2(agent)
    model.add(actions(claim("growth:Q3_vs_Q2", None, ["revenue:Q2", "revenue:Q3"], formula=GROWTH)), actions())
    agent.repair()
    assert "Answer in Portuguese." in model.calls[2].prompt  # re-derivations follow them too
    assert "Be brief." not in model.calls[2].prompt  # run instructions were for that run only


def test_tool_failures_are_logged_with_their_traceback(caplog: pytest.LogCaptureFixture) -> None:
    agent, _ = make_agent(actions({"type": "call_tool", "tool": "get_revenue", "args": {"quarter": "Q7"}}), ANALYZE)
    with caplog.at_level("WARNING", logger="corollary"):
        agent.run("t")
    (record,) = [r for r in caplog.records if "get_revenue" in r.getMessage()]
    assert record.name == "corollary.agent" and record.exc_info is not None
