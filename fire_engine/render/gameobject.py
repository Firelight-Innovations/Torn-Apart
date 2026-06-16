"""
world/gameobject.py — GameObject: the fundamental entity in Torn Apart's Unity-clone object model.

A GameObject is a container for components.  It always has exactly one
Transform.  Components define behaviour; the GameObject is just the
identity + component bag + hierarchy node.

This mirrors Unity's API shape (same method names, snake_case) but with Z-up
coordinates and batched execution semantics (see world/registry.py).

No panda3d imports — fully headless-testable.

Example
-------
    from fire_engine.render.gameobject import GameObject
    from fire_engine.render.component  import Component

    class Logger(Component):
        def start(self):
            print(f"Logger started on {self.game_object.name}")
        def update(self, dt):
            pass

    go = GameObject(name="Player", tag="player")
    go.add_component(Logger)
    # ComponentRegistry.run_frame() will call awake → start → update
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, TypeVar

from fire_engine.render.component import Component
from fire_engine.render.transform import Transform

if TYPE_CHECKING:
    pass

T = TypeVar("T", bound=Component)


class GameObject:
    """
    Container for components; the fundamental entity of the object model.

    Mirrors the Unity ``GameObject`` API in snake_case.

    Parameters
    ----------
    name  : str — human-readable name (not unique).
    tag   : str — single primary tag used by find_with_tag / compare_tag.
    layer : int — render/physics layer mask index (default 0 = Default).

    Attributes
    ----------
    id               : UUID   — unique identifier (uuid4).
    name             : str
    tag              : str
    layer            : int
    active_self      : bool   — local active flag (set via set_active).
    active_in_hierarchy : bool — True only when active_self AND all ancestors active.
    transform        : Transform — always present; cannot be removed.

    Lifecycle methods (on components) are dispatched by ComponentRegistry,
    not by GameObject itself.  GameObject calls the registry's scheduling
    methods so that awake/start are queued, not called immediately.
    """

    __slots__ = (
        "_components",
        "active_self",
        "id",
        "layer",
        "name",
        "tag",
        "transform",
    )

    def __init__(
        self,
        name: str = "GameObject",
        tag: str = "Untagged",
        layer: int = 0,
    ) -> None:
        self.id: uuid.UUID = uuid.uuid4()
        self.name: str = name
        self.tag: str = tag
        self.layer: int = layer
        self.active_self: bool = True

        self.transform: Transform = Transform()
        self.transform.game_object = self  # type: ignore[assignment]  # Transform.game_object slot starts as None; type widened by GameObject

        self._components: list[Component] = []

    # ------------------------------------------------------------------
    # active_in_hierarchy  (derived from parent chain via Transform)
    # ------------------------------------------------------------------

    @property
    def active_in_hierarchy(self) -> bool:
        """
        True when this object and all of its ancestors are active.

        Traverses the transform parent chain; O(depth) but depth is small.
        """
        if not self.active_self:
            return False
        parent_tf = self.transform.parent
        while parent_tf is not None:
            go = parent_tf.game_object
            if go is not None and not go.active_self:
                return False
            parent_tf = parent_tf.parent
        return True

    # ------------------------------------------------------------------
    # Component management
    # ------------------------------------------------------------------

    def add_component(self, t: type[T], **kwargs: object) -> T:
        """
        Construct a component of type *t*, attach it, and schedule awake/start.

        The component is constructed with **kwargs forwarded to __init__.
        Its game_object and transform back-references are set before any
        lifecycle method is called.

        Components added during run_frame are queued for the NEXT frame's
        awake/start flush (see ComponentRegistry.run_frame Known Traps).

        Parameters
        ----------
        t      : type[T] — Component subclass to add (not an instance).
        kwargs : forwarded to the component constructor.

        Returns
        -------
        T — the newly created component instance.

        Example
        -------
            from fire_engine.render.gameobject import GameObject
            go = GameObject(name="Player")
            ctrl = go.add_component(FlyController, speed=10.0)
        """
        component = t(**kwargs)
        component.game_object = self
        component.transform = self.transform
        self._components.append(component)

        # Schedule awake + start through the singleton registry.
        # Import lazily to avoid a circular import at module load time.
        from fire_engine.render.registry import ComponentRegistry

        ComponentRegistry._schedule_awake(component)

        return component

    def get_component(self, t: type[T]) -> T | None:
        """
        Return the first component of type *t* attached to this GameObject,
        or None if none is attached.

        Only exact type matches are returned (no subclasses).  Use isinstance
        if you need polymorphic lookup.

        Parameters
        ----------
        t : type[T]

        Returns
        -------
        T | None

        Example
        -------
            ctrl = go.get_component(FlyController)
            if ctrl:
                ctrl.speed = 20.0
        """
        for c in self._components:
            if type(c) is t:
                return c
        # fallback: check subclasses too (isinstance)
        for c in self._components:
            if isinstance(c, t):
                return c
        return None

    def get_components(self, t: type[T]) -> list[T]:
        """
        Return all components of type *t* attached to this GameObject.

        Parameters
        ----------
        t : type[T]

        Returns
        -------
        list[T] — may be empty.
        """
        return [c for c in self._components if isinstance(c, t)]

    def get_component_in_children(self, t: type[T]) -> T | None:
        """
        Return the first component of type *t* in this object or any
        descendant (breadth-first).

        Parameters
        ----------
        t : type[T]

        Returns
        -------
        T | None

        Example
        -------
            renderer = root.get_component_in_children(MeshRenderer)
        """
        # BFS over transform children
        queue: list[GameObject] = [self]
        while queue:
            go = queue.pop(0)
            found = go.get_component(t)
            if found is not None:
                return found
            for child_tf in go.transform.children:
                child_go = child_tf.game_object
                if child_go is not None:
                    queue.append(child_go)
        return None

    def remove_component(self, c: Component) -> None:
        """
        Detach and destroy a component.

        Calls on_disable() (if enabled) then on_destroy() at end-of-frame
        via the registry's deferred destroy queue.

        Parameters
        ----------
        c : Component — must be attached to this GameObject.

        Raises
        ------
        ValueError if the component is not attached to this object.
        """
        if c not in self._components:
            raise ValueError(f"Component {c!r} is not attached to GameObject '{self.name}'.")
        self._components.remove(c)

        # Deferred teardown via registry
        from fire_engine.render.registry import ComponentRegistry

        ComponentRegistry._schedule_destroy_component(c)

    # ------------------------------------------------------------------
    # Active state
    # ------------------------------------------------------------------

    def set_active(self, value: bool) -> None:
        """
        Set the local active flag and cascade on_enable/on_disable down the
        transform hierarchy.

        When deactivating: calls on_disable on all enabled components of this
        object and all active descendants (depth-first).
        When activating: calls on_enable on all enabled components of this
        object and all active descendants — BUT only if active_in_hierarchy
        becomes True (parent must also be active).

        Parameters
        ----------
        value : bool — True = activate, False = deactivate.

        Example
        -------
            go.set_active(False)   # disable this and all children
            go.set_active(True)    # re-enable
        """
        if self.active_self == value:
            return

        self.active_self = value

        # Walk subtree depth-first (this node first, then children)
        self._cascade_active(value)

    def _cascade_active(self, value: bool) -> None:
        """Internal: cascade enable/disable through this node and all descendants."""
        if value:
            # Only fire on_enable if active_in_hierarchy is now True
            if self.active_in_hierarchy:
                for c in self._components:
                    if c.enabled:
                        c.on_enable()
        else:
            for c in self._components:
                if c.enabled:
                    c.on_disable()

        for child_tf in self.transform.children:
            child_go = child_tf.game_object
            if child_go is not None:
                child_go._cascade_active(value)

    # ------------------------------------------------------------------
    # Tag
    # ------------------------------------------------------------------

    def compare_tag(self, tag: str) -> bool:
        """
        Return True if this object's tag matches *tag* exactly.

        Parameters
        ----------
        tag : str

        Example
        -------
            if go.compare_tag("player"):
                take_damage()
        """
        return self.tag == tag

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"GameObject(name={self.name!r}, tag={self.tag!r}, "
            f"id={str(self.id)[:8]}..., "
            f"components={[type(c).__name__ for c in self._components]})"
        )
