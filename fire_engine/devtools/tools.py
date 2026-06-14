"""
devtools/tools.py — the dev-tool plugins that populate the debug overlay.

A :class:`DevTool` is a self-contained panel: it knows how to :meth:`~DevTool.build`
its current :class:`~fire_engine.devtools.fields.Panel`, and exposes a ``revision``
the renderer watches to know when the panel's *structure* changed (so it rebuilds
widgets only then).  New tools are added by subclassing :class:`DevTool` and
registering an instance with :class:`~fire_engine.devtools.manager.DevToolsManager`
— this is the extension point the whole system is built around.

Built-in tools
--------------
- :class:`PerformanceTool` — live engine stats from injected provider callables.
- :class:`InspectorTool`   — reflected, editable view of the selected GameObject.
- :class:`ActionsTool`     — a grid of one-shot action buttons (spawn, fire event…).
- :class:`ClockTool`       — read-out of the game calendar (seed for a future
                             day/night editor).

None of this imports panda3d (hard rule 1).  Panda3D-specific values (FPS, draw
counts) arrive as plain callables supplied by the renderer in ``world/``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from fire_engine.devtools.fields import Button, Field, FieldKind, Panel, Section
from fire_engine.devtools.introspect import describe_chunk, describe_object, is_chunk

if TYPE_CHECKING:
    from fire_engine.devtools.selection import Selection


class DevTool:
    """
    Base class for every dev-overlay panel.

    Subclasses set :attr:`tool_id` / :attr:`title` and implement :meth:`build`.
    Override :attr:`revision` if the panel's structure can change at runtime
    (the default 0 means "structure is fixed; only values change").

    Attributes
    ----------
    tool_id : str — stable id; the renderer keys persistent widgets off it.
    title   : str — panel caption.
    """

    tool_id: str = "tool"
    title: str = "Tool"

    @property
    def revision(self) -> int:
        """
        Structure revision — bump when sections/fields/buttons appear or vanish.

        The renderer rebuilds its widgets whenever this value changes and only
        polls ``Field.get`` otherwise.  A fixed-structure tool can leave it at 0.
        """
        return 0

    def build(self) -> Panel:
        """Return the panel to display this frame.  Implemented by subclasses."""
        raise NotImplementedError(
            "DevTool subclasses must implement build(); see ARCHITECTURE.md §6."
        )


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


class PerformanceTool(DevTool):
    """
    Live engine performance / state read-out.

    Parameters
    ----------
    providers : dict[str, Callable[[], object]]
        Ordered mapping of label → zero-arg callable returning the current
        value (rendered via ``str``).  The renderer supplies panda3d-backed
        callables (FPS, frame ms) plus engine ones (chunk count, object count),
        so this tool stays headless.

    Example
    -------
        PerformanceTool({
            "FPS":    lambda: round(globalClock.get_average_frame_rate(), 1),
            "chunks": lambda: len(chunk_manager.chunks),
        })
    """

    tool_id = "performance"
    title = "Performance"

    def __init__(self, providers: dict[str, Callable[[], object]]) -> None:
        self._providers = dict(providers)

    def build(self) -> Panel:
        fields = [
            Field(label, FieldKind.LABEL, (lambda fn=fn: fn()))
            for label, fn in self._providers.items()
        ]
        return Panel(self.tool_id, self.title, [Section("Stats", fields)])


# ---------------------------------------------------------------------------
# Inspector
# ---------------------------------------------------------------------------


class InspectorTool(DevTool):
    """
    Editable inspector for the currently-selected GameObject.

    Rebuilds its structure (via the ``revision`` it borrows from the
    :class:`~fire_engine.devtools.selection.Selection`) whenever the selection
    changes; between changes it shows live values and applies edits straight to
    the object through :func:`~fire_engine.devtools.introspect.describe_object`.

    Parameters
    ----------
    selection : Selection — shared selection state.
    """

    tool_id = "inspector"
    title = "Inspector"

    def __init__(self, selection: Selection) -> None:
        self._selection = selection

    @property
    def revision(self) -> int:
        # Selection changes are the only thing that reshapes this panel.
        return self._selection.revision

    def build(self) -> Panel:
        go = self._selection.current
        if go is None:
            return Panel(
                self.tool_id,
                self.title,
                [
                    Section(
                        "",
                        [Field("(nothing selected)", FieldKind.LABEL, lambda: "click an object")],
                    )
                ],
                revision=self.revision,
            )
        # A picked terrain chunk is not a GameObject — route it to the chunk
        # describer (read-only voxel stats) instead of the component reflector.
        if is_chunk(go):
            title = f"Inspector — Chunk {tuple(go.coord)}"
            return Panel(self.tool_id, title, describe_chunk(go), revision=self.revision)
        title = f"Inspector — {go.name}"
        return Panel(self.tool_id, title, describe_object(go), revision=self.revision)


# ---------------------------------------------------------------------------
# Actions (spawn things / fire events)
# ---------------------------------------------------------------------------


class ActionsTool(DevTool):
    """
    A panel of one-shot action buttons — spawn props, fire events, reset state.

    Actions can be added at runtime via :meth:`add_action`; doing so bumps the
    revision so the renderer rebuilds the button row.  This is how gameplay
    systems hang their own dev verbs off the overlay without touching the
    renderer.

    Parameters
    ----------
    title : str — panel caption (default ``"Actions"``).
    actions : dict[str, Callable[[], None]] | None — initial label → handler map.

    Example
    -------
        tools = ActionsTool("World", {"Spawn Cube": spawn_cube})
        tools.add_action("Fire Explosion", explode_at_camera)
    """

    tool_id = "actions"

    def __init__(
        self,
        title: str = "Actions",
        actions: dict[str, Callable[[], None]] | None = None,
    ) -> None:
        self.title = title
        self._actions: list[tuple[str, Callable[[], None]]] = list((actions or {}).items())
        self._revision = 0

    @property
    def revision(self) -> int:
        return self._revision

    def add_action(self, label: str, handler: Callable[[], None]) -> None:
        """Append an action button and rebuild the panel next frame."""
        self._actions.append((label, handler))
        self._revision += 1

    def build(self) -> Panel:
        buttons = [Button(label, fn) for label, fn in self._actions]
        return Panel(self.tool_id, self.title, [], buttons=buttons, revision=self._revision)


# ---------------------------------------------------------------------------
# CallbackTool — ad-hoc panel from a build function (no subclass needed)
# ---------------------------------------------------------------------------


class CallbackTool(DevTool):
    """
    A panel whose contents come from a supplied ``build_fn`` each frame.

    The quickest way to surface a subsystem in the overlay without writing a
    dedicated :class:`DevTool` subclass — hand it a function that returns the
    sections/buttons.  Used (in ``world/devtools_overlay.py``) to expose the sky
    / weather / time-of-day environment controls, since that subsystem is
    panda3d-free but the binding lives in the renderer.

    Parameters
    ----------
    tool_id : str — stable id.
    title : str — panel caption.
    build_fn : Callable[[], tuple[list[Section], list[Button]]]
        Returns ``(sections, buttons)`` for the current frame.
    revision_fn : Callable[[], int] | None
        Optional structural-revision source (e.g. when the section layout can
        change).  Defaults to a constant 0 (fixed structure).

    Example
    -------
        CallbackTool("env", "Environment",
                     lambda: ([Section("Time", [...])], [Button("Noon", set_noon)]))
    """

    def __init__(
        self,
        tool_id: str,
        title: str,
        build_fn: Callable[[], tuple[list[Section], list[Button]]],
        revision_fn: Callable[[], int] | None = None,
    ) -> None:
        self.tool_id = tool_id
        self.title = title
        self._build_fn = build_fn
        self._revision_fn = revision_fn

    @property
    def revision(self) -> int:
        return self._revision_fn() if self._revision_fn is not None else 0

    def build(self) -> Panel:
        sections, buttons = self._build_fn()
        return Panel(self.tool_id, self.title, sections, buttons=buttons, revision=self.revision)


# ---------------------------------------------------------------------------
# Clock (game calendar read-out; future day/night editor lives here)
# ---------------------------------------------------------------------------


class ClockTool(DevTool):
    """
    Read-out of the game calendar — day number and time-of-day.

    This is the natural home for the upcoming day/night-cycle controls the owner
    described: when a sun-angle / time-scale system exists, add editable Fields
    here (e.g. a ``time_of_day`` slider) and they appear in the overlay with no
    renderer changes.

    Parameters
    ----------
    clock : Clock — the engine clock (duck-typed: ``game_day`` / ``game_time_of_day``).
    """

    tool_id = "clock"
    title = "Time"

    def __init__(self, clock) -> None:
        self._clock = clock

    @staticmethod
    def _fmt_tod(seconds: float) -> str:
        """Format seconds-within-day as HH:MM (24h)."""
        total_min = int(seconds // 60)
        return f"{(total_min // 60) % 24:02d}:{total_min % 60:02d}"

    def build(self) -> Panel:
        c = self._clock
        return Panel(
            self.tool_id,
            self.title,
            [
                Section(
                    "Calendar",
                    [
                        Field("day", FieldKind.LABEL, lambda: c.game_day),
                        Field(
                            "time of day",
                            FieldKind.LABEL,
                            lambda: self._fmt_tod(c.game_time_of_day),
                        ),
                    ],
                )
            ],
        )
