"""
render/sky/_impl/cover_events — shared event parsing for the rain-cover consumers.

Both :class:`~fire_engine.render.sky.rain_renderer.RainRendererComponent` and
:class:`~fire_engine.render.sky.lightning_renderer.LightningRendererComponent` own a
:class:`~fire_engine.world.terrain.RainCoverField` and mark chunk **columns** dirty
on a :class:`~fire_engine.core.event_bus.TerrainEditedEvent`, refolding a budget of
them per frame.  The event-payload parsing lived verbatim in both components; it
is hoisted here so the two render halves share one implementation (and one place
to fix if the event shape changes).

Docs: docs/systems/render.sky._impl.md
"""

from __future__ import annotations

from typing import Any

__all__ = ["edited_chunk_columns"]


def edited_chunk_columns(event: Any) -> tuple[tuple[int, int], ...]:
    """The ``(cx, cy)`` chunk columns touched by a ``TerrainEditedEvent``.

    ``event.chunk_coords`` is either a single ``(cx, cy, cz)`` flat tuple or a
    flat sequence of several such triples; this normalises both to a tuple of
    ``(cx, cy)`` columns (cz dropped — the cover folds the whole Z stack of a
    column), de-duplicated by the caller's dirty-set.

    Parameters
    ----------
    event : TerrainEditedEvent
        Carries ``chunk_coords`` (a flat ``(...,)`` of one or more ``cx,cy,cz``).

    Returns
    -------
    tuple[tuple[int, int], ...]
        The distinct-order ``(cx, cy)`` columns to mark dirty.

    Docs: docs/systems/render.sky._impl.md
    """
    raw: tuple[int, ...] = event.chunk_coords
    # A single (cx,cy,cz) flat triple vs a sequence of them: detect the
    # single-coord case (3 ints) and wrap it so the slice loop is uniform.
    if len(raw) == 3 and all(isinstance(c, int) for c in raw):
        seqs: tuple[tuple[int, ...], ...] = (raw,)
    else:
        seqs = tuple(raw[i : i + 3] for i in range(0, len(raw), 3))
    return tuple((int(c[0]), int(c[1])) for c in seqs)
