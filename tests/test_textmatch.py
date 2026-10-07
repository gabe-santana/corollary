from __future__ import annotations

import pytest

from corollary.textmatch import Figure, contains_quote, figures_in, normalize, numbers_in, value_in_text


def test_normalize() -> None:
    assert normalize("  It’s   “Q2” — up ") == 'it\'s "q2" - up'


def test_contains_quote_ignores_case_and_spacing() -> None:
    doc = "Total revenue\nfor the quarter was $4.3 billion."
    assert contains_quote(doc, "total revenue for the   quarter")
    assert not contains_quote(doc, "net revenue")
    assert not contains_quote(doc, "   ")


def test_numbers_in() -> None:
    assert numbers_in("Revenue of 4,100,000,000 grew 9.76% in Q3 of 2026, from -3.5 and 1e9") == [
        (4100000000.0, 0),
        (9.76, 2),
        (2026.0, 0),
        (-3.5, 1),
        (1e9, 0),
    ]


@pytest.mark.parametrize(
    ("value", "text", "expected"),
    [
        (4.1e9, "revenue was $4.1 billion", True),
        (4.1e9, "revenue was 4,100,000,000", True),
        (0.0976, "grew 9.76%", True),
        (9.7561, "grew 9.76%", True),
        (9.7561, "grew 9.8%", True),
        (9.7561, "grew 9.70%", False),
        (-3.5, "fell 3.5%", True),
        (4.3e9, "revenue was $4.1 billion", False),
        ("Austin", "Headquartered in austin, TX", True),
        ("Boston", "Headquartered in Austin", False),
        (True, "Flag: true", True),
        ([1], "1", False),
        # A unit fixes the scale: these used to match at some other scale.
        (4.1e9, "4 percent", False),
        (4.3e9, "revenue was $4 million", False),
        (4.3e9, "receita de 4.3 bilhões", True),
        (4.3e9, "4.3 milhões", False),
        (2.5e12, "2.5 billones", True),
        (2.5e6, "2.5 milhão", True),
        (2.5e6, "2.5 milhao", True),
        (2.5e6, "2.5 milhoes", True),
        (2.5e6, "2.5 milhões", True),
        (2.5e6, "2.5 millón", True),
        (2.5e6, "2.5 millon", True),
        (2.5e6, "2.5 millones", True),
        (2.5e9, "2.5 bilhão", True),
        (2.5e9, "2.5 bilhao", True),
        (2.5e9, "2.5 bilhoes", True),
        (2.5e12, "2.5 trilhão", True),
        (2.5e12, "2.5 trilhao", True),
        (2.5e12, "2.5 trilhoes", True),
        (2.5e12, "2.5 trilhões", True),
        (2.5e12, "2.5 billón", True),
        (2.5e12, "2.5 billon", True),
        (2.5e3, "2.5 mil", True),
        (4.3e9, "revenue was $4.3B", True),
        (4.3e9, "4,300 million", True),
        (9.76, "grew 9.76%", True),
        # Without a unit, a table header may scale the number ("in millions", "(%)").
        (4.3e9, "Revenue 4,300", True),
        (0.125, "Margin (%): 12.5", True),
        # Other percent spellings, and values stored in a smaller unit than the text uses.
        (0.0976, "up 9.76 percentage points", True),
        (0.0976, "(9.76)%", True),
        (0.0976, "9.76pp", True),
        (4300, "revenue of $4.3 billion", True),
        (4.3e12, "revenue of $4.3 billion", False),
        # A numeric string is matched as a number.
        ("4.3", "$4.3bn", True),
        ("order 1043", "order 10432", False),
        (5, ".5", False),
        (-2.5, "fell by −2.5", True),
        (False, "a falsehood", False),
        ("Austin", "Austinville", False),
    ],
)
def test_value_in_text(value: object, text: str, expected: bool) -> None:
    assert value_in_text(value, text) is expected


def test_figures_carry_the_scales_their_unit_allows() -> None:
    assert figures_in("$4.1B, 9.76 percent, 12 k, .5 and 7") == [
        Figure(4.1, 1, (1.0, 1e-3, 1e-6, 1e-9)),
        Figure(9.76, 2, (1.0, 100.0)),
        Figure(12.0, 0, (1.0, 1e-3)),
        Figure(0.5, 1),
        Figure(7.0, 0),
    ]


def test_figures_can_skip_dates_times_and_ordinals() -> None:
    text = "As of 2026-03-25 at 10:45, on 3/4/2026, ranked 15th with 42.5"
    assert [f.value for f in figures_in(text, skip_dates=True)] == [42.5]
