from __future__ import annotations

import json
from pathlib import Path

import pytest

from corollary import BeliefBase, Status, rule


@rule
def growth(previous: float, current: float) -> float:
    return (current - previous) / previous * 100


def build() -> BeliefBase:
    kb = BeliefBase()
    kb.add_document("10-K", "Revenue was $4.3 billion.")
    kb.cite("revenue:Q2", 4.3e9, document="10-K", quote="Revenue was $4.3 billion")
    kb.assert_("revenue:Q3", 4.5e9, source="tool:get_revenue")
    kb.derive("growth", growth, "revenue:Q2", "revenue:Q3")
    kb.assert_("old", 1)
    kb.retract("old", reason="obsolete")
    return kb


def test_roundtrip_preserves_graph_and_labels(tmp_path: Path) -> None:
    kb = build()
    path = tmp_path / "beliefs.json"
    kb.save(path)
    loaded = BeliefBase.load(path, rules=[growth])
    assert loaded.value("growth") == kb.value("growth")
    assert loaded.status("old") is Status.OUT
    assert loaded.why_out("old") == "retracted: obsolete"
    assert loaded.documents == kb.documents
    assert [str(e) for e in loaded.history] == [str(e) for e in kb.history]
    assert loaded.changes() == [], "loading is not a change"
    assert loaded.proof("growth").verify(loaded).ok
    assert loaded.to_dict() == kb.to_dict()


def test_loaded_base_keeps_working(tmp_path: Path) -> None:
    path = tmp_path / "beliefs.json"
    build().save(path)
    kb = BeliefBase.load(path, rules=[growth])
    kb.retract("revenue:Q2")
    kb.assert_("revenue:Q2", 4.1e9, source="tool:get_revenue")
    kb.propagate()
    assert kb.value("growth") == pytest.approx(9.75609756)
    new_ids = [j.id for j in kb.justifications("growth")]
    assert len(set(new_ids)) == len(new_ids), "justification ids continue after load"


def test_missing_rules_leave_beliefs_pending(tmp_path: Path) -> None:
    path = tmp_path / "beliefs.json"
    build().save(path)
    with pytest.warns(UserWarning, match="rules that were not passed to load"):
        kb = BeliefBase.load(path)
    kb.retract("revenue:Q3")
    kb.assert_("revenue:Q3", 5e9, source="tool:get_revenue")
    (pending,) = kb.propagate().pending
    assert "not registered" in pending.reason


def test_rejects_foreign_or_future_snapshots() -> None:
    with pytest.raises(ValueError, match="not a Corollary"):
        BeliefBase.from_dict({"format": "other"})
    data = build().to_dict()
    data["version"] = 99
    with pytest.raises(ValueError, match="newer"):
        BeliefBase.from_dict(data)


@pytest.mark.parametrize("bad_version", [True, "2", 1.5])
def test_rejects_non_integer_version_fields(bad_version: object) -> None:
    """bool is a subclass of int in Python, so True must be rejected explicitly."""
    data = build().to_dict()
    data["version"] = bad_version
    with pytest.raises(ValueError, match="must be an integer"):
        BeliefBase.from_dict(data)


def test_rejects_negative_version() -> None:
    data = build().to_dict()
    data["version"] = -1
    with pytest.raises(ValueError, match="invalid"):
        BeliefBase.from_dict(data)


def test_missing_version_still_loads_as_the_oldest_format() -> None:
    """Unchanged behavior: a snapshot with no version field at all predates versioning."""
    data = build().to_dict()
    del data["version"]
    BeliefBase.from_dict(data, rules=[growth])  # must not raise


@pytest.mark.parametrize("version", [0, 1, 2])
def test_every_real_version_still_loads(version: int) -> None:
    data = build().to_dict()
    data["version"] = version
    BeliefBase.from_dict(data, rules=[growth])  # must not raise


def test_snapshot_is_plain_json() -> None:
    text = json.dumps(build().to_dict())
    assert '"format": "corollary.beliefbase"' in text


def test_corrupt_snapshots_raise_value_error() -> None:
    data = build().to_dict()
    del data["beliefs"][0]["belief"]
    with pytest.raises(ValueError, match="corrupt Corollary snapshot"):
        BeliefBase.from_dict(data)


def test_save_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "beliefs.json"
    build().save(path)
    before = path.read_text(encoding="utf-8")
    kb = build()
    kb.assert_("more", 1, source="tool:x")

    def crash(fd: int) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("corollary._io.os.fsync", crash)  # fail mid-write
    with pytest.raises(OSError, match="disk full"):
        kb.save(path)
    assert path.read_text(encoding="utf-8") == before
    assert [p.name for p in tmp_path.iterdir()] == ["beliefs.json"]  # no temporary file left behind


def test_save_works_while_another_handle_reads_the_file(tmp_path: Path) -> None:
    path = tmp_path / "beliefs.json"
    build().save(path)
    with path.open(encoding="utf-8"):  # on Windows, this blocks os.replace
        build().save(path)
    assert BeliefBase.load(path, rules=[growth]).value("growth") is not None


def test_load_reports_the_callers_mistakes_as_such(tmp_path: Path) -> None:
    path = tmp_path / "beliefs.json"
    build().save(path)
    with pytest.raises(AttributeError):
        BeliefBase.load(path, rules=[lambda x: x])  # not a Rule: the caller's mistake, not corruption


def test_missing_rule_warning_points_at_the_caller(tmp_path: Path) -> None:
    path = tmp_path / "beliefs.json"
    build().save(path)
    with pytest.warns(UserWarning) as caught:
        BeliefBase.load(path)
    assert caught[0].filename == __file__
