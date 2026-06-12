"""Editor undo/redo command stack (EDITOR_PRD §5.4, Phase E3).

Editor-side (not an engine concern): a bounded stack of reversible edits.
Terrain edits (:class:`EditCommand`) snapshot the pre- and post-edit material
arrays of the chunks they touched; undo writes the ``before`` snapshot back,
redo writes ``after``. Brushes are local, so a command holds only a handful of
32 KB chunk arrays. Scene ops (:class:`SceneCommand`) snapshot the whole
authoring hierarchy delta — scenes are tiny dicts, so full snapshots are cheap.
Both command types share ONE stack, so Ctrl+Z walks terrain and scene edits in
true chronological order.

This module is pure data + numpy; applying a command back onto the live session
(remesh/relight/restream or scene apply_delta) is the caller's job (see
``services/chunks.py``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Cap the history depth. Each chunk snapshot is chunk_size**3 bytes (32 KB at
# 32**3 uint8); a brush touches a few chunks, so ~200 entries stays well under
# the §5.4 256 MB budget. Oldest entries drop first (LRU).
MAX_HISTORY: int = 200


@dataclass
class EditCommand:
    """One reversible terrain edit.

    Attributes:
        label: Human-readable description (e.g. ``"sphere remove"``).
        before: ``{coord: uint8 materials}`` snapshot taken *before* the edit.
        after: ``{coord: uint8 materials}`` snapshot taken *after* the edit.
    """

    label: str
    before: dict[tuple[int, int, int], np.ndarray]
    after: dict[tuple[int, int, int], np.ndarray]

    @property
    def coords(self) -> list[tuple[int, int, int]]:
        """Chunk coords this command affects."""
        return list(self.after.keys())


@dataclass
class SceneCommand:
    """One reversible scene-hierarchy edit (create/rename/reparent/transform/delete).

    Attributes:
        label: Human-readable description (e.g. ``"transform 7"``, ``"create cube"``).
        before_delta: Full ``SceneObjectStore.get_delta()`` snapshot pre-edit.
        after_delta: Full snapshot post-edit.
        timestamp: ``time.monotonic()`` at push — used to coalesce rapid
            same-label edits (a throttled gizmo drag becomes one undo step).
    """

    label: str
    before_delta: dict
    after_delta: dict
    timestamp: float


class UndoStack:
    """Bounded undo/redo stack of :class:`EditCommand` / :class:`SceneCommand`.

    Pushing a new command clears the redo stack (standard editor semantics).

    Example::

        stack = UndoStack()
        stack.push(EditCommand("sphere remove", before, after))
        cmd = stack.undo()   # returns the command to revert (write its `before`)
        cmd = stack.redo()   # returns the command to re-apply (write its `after`)
    """

    def __init__(self, max_history: int = MAX_HISTORY) -> None:
        self._undo: list = []
        self._redo: list = []
        self._max = max_history

    def peek(self):
        """The most recent undoable command, or ``None`` (no state change)."""
        return self._undo[-1] if self._undo else None

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def push(self, command: "EditCommand | SceneCommand") -> None:
        """Record a freshly applied command; clears the redo history."""
        self._undo.append(command)
        self._redo.clear()
        if len(self._undo) > self._max:
            self._undo.pop(0)  # LRU-drop the oldest

    def replace_top(self, command: "EditCommand | SceneCommand") -> None:
        """Swap the most recent command in place (coalescing); requires one."""
        self._undo[-1] = command

    def undo(self) -> "EditCommand | SceneCommand | None":
        """Pop the last command to the redo stack and return it (or ``None``)."""
        if not self._undo:
            return None
        cmd = self._undo.pop()
        self._redo.append(cmd)
        return cmd

    def redo(self) -> "EditCommand | SceneCommand | None":
        """Pop the last undone command back to the undo stack and return it."""
        if not self._redo:
            return None
        cmd = self._redo.pop()
        self._undo.append(cmd)
        return cmd

    def clear(self) -> None:
        """Drop all history (e.g. on ``world.open``)."""
        self._undo.clear()
        self._redo.clear()
