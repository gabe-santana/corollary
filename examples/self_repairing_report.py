"""A 20-conclusion financial report that repairs itself when one input is corrected.

Run it with no API key and no network:

    python examples/self_repairing_report.py

Every conclusion is derived by a deterministic rule, so the whole repair is pure kernel work:
retract one premise, assert the corrected value, and propagate. Only the conclusions that
actually depend on the corrected figure are recomputed, and the cascade stops wherever a
recomputed value turns out unchanged.
"""

from __future__ import annotations

from collections import Counter
from statistics import mean

from corollary import BeliefBase, ChangeKind, rule

evaluations: Counter[str] = Counter()


def counted(fn):  # type: ignore[no-untyped-def]
    """Count rule evaluations so the demo can show how little work a repair does."""

    def wrapper(*args):  # type: ignore[no-untyped-def]
        evaluations[fn.__name__] += 1
        return fn(*args)

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


@rule
@counted
def growth(previous: float, current: float) -> float:
    """Quarter-over-quarter growth, in percent."""
    return (current - previous) / previous * 100


@rule
@counted
def gross_margin(revenue: float, cogs: float) -> float:
    """Gross margin, in percent."""
    return (revenue - cogs) / revenue * 100


@rule
@counted
def total(*values: float) -> float:
    return sum(values)


@rule
@counted
def difference(a: float, b: float) -> float:
    return a - b


@rule
@counted
def ratio_pct(part: float, whole: float) -> float:
    return part / whole * 100


@rule
@counted
def ratio(a: float, b: float) -> float:
    return a / b


@rule
@counted
def average(*values: float) -> float:
    return mean(values)


@rule
@counted
def growth_trend(avg_growth: float) -> str:
    return "strong" if avg_growth >= 8 else "modest" if avg_growth >= 3 else "flat"


@rule
@counted
def margin_trend(first: float, last: float) -> str:
    return "expanding" if last > first + 0.5 else "contracting" if last < first - 0.5 else "stable"


@rule
@counted
def fx_risk(exposure: float) -> bool:
    """More than a quarter of revenue is earned in foreign currency."""
    return exposure > 0.25


@rule
@counted
def leverage_risk(leverage: float) -> bool:
    """Debt above three years of operating income."""
    return leverage > 3


@rule
@counted
def outlook(growth: str, margins: str, fx: bool, levered: bool) -> str:
    risks = int(fx) + int(levered)
    if growth == "flat" or margins == "contracting" or risks == 2:
        return "negative"
    if growth == "strong" and margins == "expanding" and risks == 0:
        return "positive"
    return "mixed"


def build() -> BeliefBase:
    kb = BeliefBase()
    filings = "tool:sec_filings"
    revenue = {"Q1": 4.0e9, "Q2": 4.3e9, "Q3": 4.5e9, "Q4": 4.8e9}
    cogs = {"Q1": 2.4e9, "Q2": 2.55e9, "Q3": 2.6e9, "Q4": 2.7e9}
    for q, value in revenue.items():
        kb.assert_(f"revenue:{q}", value, source=filings, claim=f"{q} revenue")
        kb.assert_(f"cogs:{q}", cogs[q], source=filings, claim=f"{q} cost of goods sold")
    kb.assert_("opex:FY", 4.2e9, source=filings, claim="Operating expenses for the year")
    kb.assert_("debt", 9.0e9, source=filings, claim="Total debt")
    kb.assert_("fx:exposure", 0.31, source="tool:treasury", claim="Share of revenue in foreign currency")

    # 20 conclusions
    for prev, cur in (("Q1", "Q2"), ("Q2", "Q3"), ("Q3", "Q4")):
        kb.derive(f"growth:{cur}", growth, f"revenue:{prev}", f"revenue:{cur}")
    for q in revenue:
        kb.derive(f"gross_margin:{q}", gross_margin, f"revenue:{q}", f"cogs:{q}")
    kb.derive("revenue:FY", total, *(f"revenue:{q}" for q in revenue))
    kb.derive("cogs:FY", total, *(f"cogs:{q}" for q in revenue))
    kb.derive("gross_profit:FY", difference, "revenue:FY", "cogs:FY")
    kb.derive("gross_margin:FY", ratio_pct, "gross_profit:FY", "revenue:FY")
    kb.derive("operating_income:FY", difference, "gross_profit:FY", "opex:FY")
    kb.derive("operating_margin:FY", ratio_pct, "operating_income:FY", "revenue:FY")
    kb.derive("leverage", ratio, "debt", "operating_income:FY")
    kb.derive("growth:avg", average, "growth:Q2", "growth:Q3", "growth:Q4")
    kb.derive("trend:growth", growth_trend, "growth:avg")
    kb.derive("trend:margins", margin_trend, "gross_margin:Q1", "gross_margin:Q4")
    kb.derive("risk:fx", fx_risk, "fx:exposure")
    kb.derive("risk:leverage", leverage_risk, "leverage")
    kb.derive("outlook", outlook, "trend:growth", "trend:margins", "risk:fx", "risk:leverage")
    kb.changes()
    return kb


def main() -> None:
    kb = build()
    conclusions = [b for b in kb if kb.support(b.ref) and not kb.support(b.ref).is_premise]  # type: ignore[union-attr]
    print(f"Built a report with {len(conclusions)} conclusions from {len(kb) - len(conclusions)} premises.")
    print(f"Outlook: {kb.value('outlook')}  (growth {kb.value('trend:growth')}, margins {kb.value('trend:margins')})")
    print()

    print("Q2 revenue is restated from $4.3B to $4.1B in a 10-K/A.\n")
    evaluations.clear()
    kb.correct("revenue:Q2", 4.1e9, reason="restated in 10-K/A", claim="Q2 revenue (restated)")
    result = kb.propagate(include_kept=True)

    for change in result:
        if change.kind is not ChangeKind.KEPT and not change.key.startswith("revenue:Q2"):
            print(change)
    print()
    for change in result.kept:
        print(change)

    rederived = {c.key for c in result.added} - {"revenue:Q2"}
    cut_off = [b.key for b in result.rederived if b.revision == 1]
    print()
    print(
        f"{len(rederived)} conclusions changed, {len(result.kept)} were untouched, "
        f"{len(cut_off)} recomputed but unchanged, which stopped the cascade ({', '.join(cut_off)})."
    )
    print(f"Rule evaluations during the repair: {sum(evaluations.values())}. Model calls: 0.")
    print()
    print(f"Outlook is now: {kb.value('outlook')}. Why?")
    print(kb.proof("risk:leverage").render(show_sources=False, ascii=True))
    print()
    print(kb.proof("outlook").verify(kb))


if __name__ == "__main__":
    main()
