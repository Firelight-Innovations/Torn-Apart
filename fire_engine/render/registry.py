"""
world/registry.py — ComponentRegistry: the batched Unity-order executor.

The registry is a module-level singleton that owns all component lifecycle
dispatch.  It maintains per-type buckets and two queues (pending-awake,
pending-start) that are flushed in strict Unity order each frame.

Execution order per frame (run_frame):
  1. Flush pending-awake queue:
       for each component: awake() → (if enabled) on_enable()
  2. Flush pending-start queue:
       for each component: (if not started) start(); mark _started = True
  3. update(dt) — per type bucket, only enabled + active_in_hierarchy
  4. late_update(dt) — per type bucket, same guard
  5. fixed_update(fixed_dt) — driven by Clock.fixed_steps() (0–5× per frame)
  6. Flush deferred-destroy queue (end of frame):
       on_disable() (if enabled) → on_destroy()

Components added DURING iteration (e.g. inside update) are placed in the
NEXT frame's pending-awake queue, not the current one — this matches Unity
semantics and avoids iterator mutation bugs.  Implementation: copy the queue
snapshot before iterating (see Known Traps in DEVELOPMENT_PLAN.md).

Module-level functions mirror Unity statics:
  instantiate(template, position, rotation, parent) -> GameObject
  destroy(obj_or_component, delay)
  find_with_tag(tag) -> GameObject | None
  find_objects_with_tag(tag) -> list[GameObject]

No panda3d imports — fully headless-testable.

Example
-------
    from fire_engine.render.registry import ComponentRegistry, instantiate
    from fire_engine.render.gameobject import GameObject
    from fire_engine.render.component import Component
    from fire_engine.core.clock import Clock
    from fire_engine.core.event_bus import EventBus

    clock = Clock(fixed_dt=0.02, bus=EventBus())

    class Ticker(Component):
        def __init__(self):
            super().__init__()
            self.ticks = 0
        def update(self, dt):
            self.ticks += 1

    go = instantiate()
    t  = go.add_component(Ticker)

    ComponentRegistry.run_frame(clock)  # awake + start + update
    assert t.ticks == 1

    ComponentRegistry.clear()           # reset for next test

Docs: docs/systems/render.md
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, TypeVar

from fire_engine.core.math3d import Quat, Vec3
from fire_engine.core.profiler import get_profiler

# Cache of (phase, component_type) -> compound scope name, so the per-frame hot
# path builds no strings after the first sight of a type.  Names look like
# "Update:WeatherMapComponent" so the profiler/overlay breakdown attributes
# cost to the exact component bucket (the weather system shows up by name).
_SCOPE_NAME_CACHE: dict[tuple[str, type], str] = {}


def _scope_name(phase: str, t: type) -> str:
    key = (phase, t)
    name = _SCOPE_NAME_CACHE.get(key)
    if name is None:
        name = _SCOPE_NAME_CACHE[key] = f"{phase}:{t.__name__}"
    return name


if TYPE_CHECKING:
    from fire_engine.core.clock import Clock
    from fire_engine.render.component import Component
    from fire_engine.render.gameobject import GameObject
    from fire_engine.render.transform import Transform

T = TypeVar("T")


# _RegistryState — all mutable singleton state in one object (easy to clear)


class _RegistryState:
    """Internal mutable state bag for ComponentRegistry."""

    def __init__(self) -> None:
        # All live GameObjects (for tag lookups)
        self.objects: list[GameObject] = []

        # Per-type buckets: {component_type: [component, ...]}
        # Ordering within a bucket mirrors insertion order.
        self.buckets: dict[type, list[Component]] = defaultdict(list)

        # Pending lifecycle queues
        self.pending_awake: list[Component] = []
        self.pending_start: list[Component] = []

        # Deferred destroy queues (flushed at end of frame)
        self.pending_destroy_components: list[Component] = []
        self.pending_destroy_objects: list[GameObject] = []


_STATE: _RegistryState = _RegistryState()


# ComponentRegistry — public singleton facade


class _ComponentRegistry:
    """
    Singleton facade for all component lifecycle dispatch.

    Use the module-level ``ComponentRegistry`` instance rather than
    instantiating this class.

    Thread-safety: single-threaded only (Python GIL keeps frame loop safe).
    """

    # Internal scheduling (called by GameObject)

    def _schedule_awake(self, component: Component) -> None:
        """Add a newly created component to the pending-awake queue."""
        _STATE.pending_awake.append(component)
        # Register in the appropriate type bucket
        t = type(component)
        _STATE.buckets[t].append(component)
        # Register the owning GameObject for tag lookups
        go = component.game_object
        if go is not None and go not in _STATE.objects:
            _STATE.objects.append(go)

    def _schedule_destroy_component(self, component: Component) -> None:
        """Schedule a component for end-of-frame teardown."""
        _STATE.pending_destroy_components.append(component)

    # Main run_frame

    def run_frame(self, clock: Clock) -> None:
        """
        Drive one full frame of component lifecycle dispatch.

        Call once per frame from App (or tests).

        Parameters
        ----------
        clock : Clock — used for dt (real frame) and fixed_steps() iteration.

        Dispatch order
        --------------
        1. ALL pending awake() + on_enable()   (snapshot before iterating)
        2. ALL pending start()                 (snapshot before iterating)
        3. update(dt)    — per type bucket, enabled + active_in_hierarchy only
        4. late_update(dt) — per type bucket, same guard
        5. fixed_update(fixed_dt) × N          — driven by clock.fixed_steps()
        6. Deferred destroy flush               — on_disable + on_destroy
        """
        dt = clock.dt

        self._flush_awake_and_start()

        # -- 3–4. update / late_update -------------------------------------
        self._dispatch_phase("Update", dt)
        self._dispatch_phase("LateUpdate", dt)

        # -- 5. fixed_update (driven by clock accumulator) -----------------
        for fixed_dt in clock.fixed_steps():
            self._dispatch_phase("FixedUpdate", fixed_dt)

        # -- 6. Deferred destroy -------------------------------------------
        self._flush_destroy()

    def _flush_awake_and_start(self) -> None:
        """Flush pending-awake and pending-start queues (steps 1–2 of run_frame)."""
        # -- 1. Awake + on_enable -----------------------------------------
        # Snapshot the queue so components added inside awake() go to NEXT frame.
        awake_batch = _STATE.pending_awake[:]
        _STATE.pending_awake.clear()

        for comp in awake_batch:
            comp.awake()
            if comp.enabled:
                comp.on_enable()
            # Move to pending-start
            _STATE.pending_start.append(comp)

        # -- 2. Start ------------------------------------------------------
        # Snapshot so components added inside start() go to NEXT frame.
        start_batch = _STATE.pending_start[:]
        _STATE.pending_start.clear()

        for comp in start_batch:
            if not comp._started:
                comp.start()
                comp._started = True

    def _dispatch_phase(self, phase: str, dt: float) -> None:
        """
        Dispatch one lifecycle phase to every active component in each bucket.

        One profiler scope per component TYPE bucket so the per-frame breakdown
        shows exactly which component kind is hot (e.g. "Update:WeatherMapComponent").
        scope() is a no-op when the profiler is disabled, costing only a bool check
        and a cached dict lookup.

        Parameters
        ----------
        phase : str — lifecycle method name ("Update", "LateUpdate", "FixedUpdate").
        dt    : float — delta time in seconds for this phase.
        """
        prof = get_profiler()
        for t, bucket in list(_STATE.buckets.items()):
            bucket_snap = bucket[:]
            with prof.scope(_scope_name(phase, t)):
                for comp in bucket_snap:
                    if self._is_active(comp):
                        self._call_phase(phase, comp, dt)

    @staticmethod
    def _call_phase(phase: str, comp: Component, dt: float) -> None:
        """Call the appropriate lifecycle method on *comp* for *phase*."""
        if phase == "Update":
            comp.update(dt)
        elif phase == "LateUpdate":
            comp.late_update(dt)
        else:
            comp.fixed_update(dt)

    def _is_active(self, comp: Component) -> bool:
        """Return True if the component should tick this frame."""
        if not comp.enabled:
            return False
        go = comp.game_object
        if go is None:
            return False
        return go.active_in_hierarchy

    def _flush_destroy(self) -> None:
        """Execute all pending end-of-frame destroy operations."""
        # Component-level destroys
        comp_batch = _STATE.pending_destroy_components[:]
        _STATE.pending_destroy_components.clear()
        for comp in comp_batch:
            if comp.enabled:
                comp.on_disable()
            comp.on_destroy()
            # Remove from bucket
            t = type(comp)
            if t in _STATE.buckets and comp in _STATE.buckets[t]:
                _STATE.buckets[t].remove(comp)

        # GameObject-level destroys
        obj_batch = _STATE.pending_destroy_objects[:]
        _STATE.pending_destroy_objects.clear()
        for go in obj_batch:
            # Destroy all components
            for comp in list(go._components):
                if comp.enabled:
                    comp.on_disable()
                comp.on_destroy()
                t = type(comp)
                if t in _STATE.buckets and comp in _STATE.buckets[t]:
                    _STATE.buckets[t].remove(comp)
            go._components.clear()
            # Remove from objects list
            if go in _STATE.objects:
                _STATE.objects.remove(go)

    # Module-level Unity statics

    def instantiate(
        self,
        template: GameObject | None = None,
        position: Vec3 | None = None,
        rotation: Quat | None = None,
        parent: Transform | None = None,
    ) -> GameObject:
        """
        Create (and register) a new GameObject, optionally copying a template.

        Parameters
        ----------
        template : GameObject | None — if provided, copies name/tag/layer;
                    components are NOT deep-copied (shallow identity copy only).
        position : Vec3  — initial world-space position in meters (default ZERO).
        rotation : Quat  — initial world-space rotation (default identity).
        parent   : Transform | None — parent transform.

        Returns
        -------
        GameObject — freshly created, registered, with awake queued.

        Example
        -------
            from fire_engine.render.registry import instantiate
            from fire_engine.core.math3d import Vec3

            bullet = instantiate(position=Vec3(0, 10, 0))
        """
        from fire_engine.render.gameobject import GameObject

        if position is None:
            position = Vec3(0.0, 0.0, 0.0)
        if rotation is None:
            rotation = Quat.identity()

        if template is not None:
            go = GameObject(name=template.name, tag=template.tag, layer=template.layer)
        else:
            go = GameObject()

        go.transform.local_position = position
        go.transform.local_rotation = rotation

        if parent is not None:
            go.transform.set_parent(parent, keep_world=False)

        # Register (even without components)
        if go not in _STATE.objects:
            _STATE.objects.append(go)

        return go

    def destroy(
        self,
        obj_or_component: GameObject | Component,
        delay: float = 0.0,
    ) -> None:
        """
        Schedule destruction of a GameObject or Component.

        Deferred to end-of-frame (delay parameter is accepted but not yet
        implemented for non-zero values; deferred to the current frame's
        end-of-frame flush in all cases for Session 1).

        Parameters
        ----------
        obj_or_component : GameObject | Component
        delay            : float — seconds to wait (0 = end of current frame).

        Note
        ----
        Non-zero delay is not implemented in Session 1.  All destroys are
        deferred to the current frame's flush regardless of the delay value.

        Example
        -------
            destroy(bullet_go)           # destroyed at end of this frame
            destroy(old_component)       # component torn down at end of frame
        """
        from fire_engine.render.component import Component
        from fire_engine.render.gameobject import GameObject

        if isinstance(obj_or_component, GameObject):
            if obj_or_component not in _STATE.pending_destroy_objects:
                _STATE.pending_destroy_objects.append(obj_or_component)
        elif isinstance(obj_or_component, Component):
            if obj_or_component not in _STATE.pending_destroy_components:
                _STATE.pending_destroy_components.append(obj_or_component)
        else:
            raise TypeError(
                f"destroy() expects a GameObject or Component, got {type(obj_or_component)}"
            )

    def find_with_tag(self, tag: str) -> GameObject | None:
        """
        Return the first registered GameObject whose tag matches *tag*, or None.

        Parameters
        ----------
        tag : str — exact tag string (case-sensitive).

        Returns
        -------
        GameObject | None

        Example
        -------
            player = find_with_tag("player")
        """
        for go in _STATE.objects:
            if go.tag == tag:
                return go
        return None

    def find_objects_with_tag(self, tag: str) -> list[GameObject]:
        """
        Return all registered GameObjects whose tag matches *tag*.

        Parameters
        ----------
        tag : str — exact tag string (case-sensitive).

        Returns
        -------
        list[GameObject] — may be empty.

        Example
        -------
            enemies = find_objects_with_tag("enemy")
        """
        return [go for go in _STATE.objects if go.tag == tag]

    # Test isolation

    def clear(self) -> None:
        """
        Reset all registry state.

        Call between tests to prevent component/object leakage.

        Example
        -------
            ComponentRegistry.clear()
        """
        global _STATE
        _STATE = _RegistryState()


# Module-level singleton

ComponentRegistry: _ComponentRegistry = _ComponentRegistry()

# Convenience module-level functions (mirrors Unity statics and ARCHITECTURE §5.4)


def instantiate(
    template: GameObject | None = None,
    position: Vec3 | None = None,
    rotation: Quat | None = None,
    parent: Transform | None = None,
) -> GameObject:
    """
    Create a new GameObject at *position* with *rotation*.

    Delegates to ComponentRegistry.instantiate.  See that method for full
    parameter documentation.

    Returns
    -------
    GameObject

    Docs: docs/systems/render.md
    """
    return ComponentRegistry.instantiate(
        template=template,
        position=position,
        rotation=rotation,
        parent=parent,
    )


def destroy(obj_or_component: GameObject | Component, delay: float = 0.0) -> None:
    """
    Schedule destruction of a GameObject or Component at end of frame.

    Delegates to ComponentRegistry.destroy.

    Docs: docs/systems/render.md
    """
    ComponentRegistry.destroy(obj_or_component, delay=delay)


def find_with_tag(tag: str) -> GameObject | None:
    """Return the first registered GameObject with the given tag, or None.

    Docs: docs/systems/render.md
    """
    return ComponentRegistry.find_with_tag(tag)


def find_objects_with_tag(tag: str) -> list[GameObject]:
    """Return all registered GameObjects with the given tag.

    Docs: docs/systems/render.md
    """
    return ComponentRegistry.find_objects_with_tag(tag)
