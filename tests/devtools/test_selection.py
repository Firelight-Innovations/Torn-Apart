"""
tests/devtools/test_selection.py — tests for fire_engine/devtools/selection.py.

Covers Selection: initial state, revision counter, set/clear semantics,
no-op on same object, and on_change listener. Fully headless; no panda3d imports.
"""

from __future__ import annotations

from fire_engine.devtools.selection import Selection

# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def test_selection_initial_state():
    sel = Selection()
    assert sel.current is None
    assert sel.revision == 0


# ---------------------------------------------------------------------------
# set / revision
# ---------------------------------------------------------------------------


def test_set_changes_current_and_bumps_revision():
    sel = Selection()
    obj = object()
    sel.set(obj)  # type: ignore[arg-type]
    assert sel.current is obj
    assert sel.revision == 1


def test_set_same_object_is_noop():
    sel = Selection()
    obj = object()
    sel.set(obj)  # type: ignore[arg-type]
    rev_before = sel.revision
    sel.set(obj)  # type: ignore[arg-type]
    assert sel.revision == rev_before


def test_set_different_objects_bumps_revision_each_time():
    sel = Selection()
    a, b, c = object(), object(), object()
    sel.set(a)  # type: ignore[arg-type]
    sel.set(b)  # type: ignore[arg-type]
    sel.set(c)  # type: ignore[arg-type]
    assert sel.revision == 3
    assert sel.current is c


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


def test_clear_deselects_and_bumps_revision():
    sel = Selection()
    sel.set(object())  # type: ignore[arg-type]
    r0 = sel.revision
    sel.clear()
    assert sel.current is None
    assert sel.revision == r0 + 1


def test_clear_when_already_none_is_noop():
    # clear() calls set(None); _current is already None → same object → no-op
    sel = Selection()
    sel.clear()
    assert sel.revision == 0  # no change: None is None


# ---------------------------------------------------------------------------
# on_change listener
# ---------------------------------------------------------------------------


def test_on_change_listener_fires_on_set():
    sel = Selection()
    seen = []
    sel.on_change(lambda go: seen.append(go))
    obj = object()
    sel.set(obj)  # type: ignore[arg-type]
    assert seen == [obj]


def test_on_change_listener_fires_on_clear():
    sel = Selection()
    seen = []
    sel.on_change(lambda go: seen.append(go))
    sel.set(object())  # type: ignore[arg-type]
    seen.clear()
    sel.clear()
    assert seen == [None]


def test_on_change_listener_not_fired_on_noop():
    sel = Selection()
    seen = []
    sel.on_change(lambda go: seen.append(go))
    obj = object()
    sel.set(obj)  # type: ignore[arg-type]
    seen.clear()
    sel.set(obj)  # same object → no-op
    assert seen == []


def test_multiple_listeners_all_fired():
    sel = Selection()
    log1, log2 = [], []
    sel.on_change(lambda go: log1.append(go))
    sel.on_change(lambda go: log2.append(go))
    obj = object()
    sel.set(obj)  # type: ignore[arg-type]
    assert log1 == [obj]
    assert log2 == [obj]


# ---------------------------------------------------------------------------
# revision is monotonically increasing
# ---------------------------------------------------------------------------


def test_revision_never_decreases():
    sel = Selection()
    revisions = [sel.revision]
    for _ in range(5):
        sel.set(object())  # type: ignore[arg-type]
        revisions.append(sel.revision)
    for i in range(1, len(revisions)):
        assert revisions[i] >= revisions[i - 1]
