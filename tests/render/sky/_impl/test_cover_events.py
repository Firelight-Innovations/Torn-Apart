"""Headless tests for ``render.sky._impl.cover_events.edited_chunk_columns``.

The helper is panda3d-free (it only reads ``event.chunk_coords``), so it runs in
the headless suite.  Covered: single-triple payloads, multi-triple payloads, the
cz axis being dropped, and column order preserved.
"""

from __future__ import annotations

from dataclasses import dataclass

from fire_engine.render.sky._impl.cover_events import edited_chunk_columns


@dataclass(frozen=True)
class _FakeEdit:
    """Stand-in for ``TerrainEditedEvent`` — only ``chunk_coords`` is read."""

    chunk_coords: tuple[int, ...]


def test_single_triple_yields_one_column() -> None:
    cols = edited_chunk_columns(_FakeEdit((3, -4, 7)))
    assert cols == ((3, -4),)


def test_drops_the_cz_axis() -> None:
    # Same (cx, cy) at two different cz layers collapse to the same column.
    cols = edited_chunk_columns(_FakeEdit((2, 5, 0, 2, 5, 1)))
    assert cols == ((2, 5), (2, 5))


def test_multiple_triples_preserve_order() -> None:
    cols = edited_chunk_columns(_FakeEdit((0, 0, 0, 1, 2, 0, -3, 4, 9)))
    assert cols == ((0, 0), (1, 2), (-3, 4))


def test_empty_payload_yields_nothing() -> None:
    assert edited_chunk_columns(_FakeEdit(())) == ()


def test_result_is_hashable_for_a_dirty_set() -> None:
    # The components feed the result straight into a set[tuple[int, int]].
    dirty: set[tuple[int, int]] = set()
    dirty.update(edited_chunk_columns(_FakeEdit((1, 1, 0, 1, 1, 2))))
    assert dirty == {(1, 1)}
