"""
save/saveable.py — Saveable protocol and SaveIncompatibleError.

Defines the ``Saveable`` structural interface that any system must implement
to participate in delta saves.  The protocol is **exactly** as specified in
ARCHITECTURE.md §5.12: each system exposes a ``save_key`` attribute and two
methods that reduce its mutable state to a plain dict of primitives / numpy
arrays (get_delta) and restore from such a dict after baseline regeneration
(apply_delta).

The save format is seed-based:
    world = regenerate_from_seed() + apply_delta(saved_delta)

That means saves store only *deviations from the procedurally determined
baseline*, so an untouched world costs ~0 bytes.  Hard Rule 3: no pickle
anywhere — deltas are plain dicts of primitives / numpy arrays only, no live
object references.

Example
-------
    from torn_apart.save.saveable import Saveable

    class MySystem:
        save_key: str = "my_system"

        def get_delta(self) -> dict:
            # Return only what deviates from the procedural baseline.
            return {"health": 42}

        def apply_delta(self, delta: dict) -> None:
            # Re-derive baseline from seed first; then overlay delta.
            self.health = delta.get("health", self._baseline_health)

    # Runtime check:
    assert isinstance(MySystem(), Saveable)   # True at runtime
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Saveable(Protocol):
    """
    Structural interface for systems that participate in delta saves.

    All systems that wish to have their state persisted must implement this
    protocol.  ``SaveManager`` never imports the concrete system types — they
    register themselves into the manager at boot (inversion of control), so
    adding a new system never touches save code.

    Attributes
    ----------
    save_key : str
        Unique identifier for this system's save blob (e.g. ``"terrain"``,
        ``"ai"``, ``"economy"``).  Used as the dict key in the on-disk envelope.
        **Must be a non-empty ASCII string with no spaces.**

    Notes
    -----
    - ``get_delta`` must return **only** plain Python primitives (int, float,
      str, bool, None, list, dict) or ``numpy.ndarray`` values.  No live object
      references, no class instances, no pickle (Hard Rule 3).
    - ``apply_delta`` is called *after* baseline regeneration from seed.  The
      system should first produce its deterministic baseline (from the world
      seed) and then overlay the delta on top.
    - The order in which ``apply_delta`` is called across registered saveables
      is the registration order (``SaveManager.register`` call order).  Per
      ARCHITECTURE.md §4a.4 sequence diagram, terrain is applied before AI,
      which is applied before economy/politics.
    """

    save_key: str

    def get_delta(self) -> dict:
        """
        Return this system's deviations from its procedural baseline.

        Only data that differs from what the world seed would deterministically
        regenerate should be included.  An unmodified system should return an
        empty dict ``{}``.

        Returns
        -------
        dict
            Plain dict of primitives / numpy arrays.  Keys must be serialisable
            to msgpack (strings, ints, floats, or tuples of those — tuple keys
            are encoded specially by SaveManager).
        """
        ...  # pragma: no cover

    def apply_delta(self, delta: dict) -> None:
        """
        Overlay saved delta onto the already-regenerated procedural baseline.

        Called by ``SaveManager.load`` in registration order after the world
        seed has been used to regenerate the baseline state.

        Parameters
        ----------
        delta : dict
            As produced by :meth:`get_delta` (decoded from the save file).
            If the system's save_key was absent from the save file, this method
            is not called at all — the system retains its fresh-generated state.
        """
        ...  # pragma: no cover


class SaveIncompatibleError(Exception):
    """
    Raised by ``SaveManager.load`` when a save file cannot be loaded safely.

    Triggers:
    - ``format_version`` in the file is newer than the engine supports.
    - ``world_seed`` in the save header does not match the current
      ``Config.world_seed``.
    - ``config_digest`` in the save header does not match the digest of the
      current ``Config``.

    When this exception is raised, **no partial load has occurred**: the
    engine state is unchanged (the save file was validated before any
    ``apply_delta`` calls).

    Attributes
    ----------
    message : str
        Human-readable explanation of the incompatibility.

    Example
    -------
        from torn_apart.save import SaveIncompatibleError

        try:
            save_manager.load("saves/quick.ta")
        except SaveIncompatibleError as exc:
            print(f"Cannot load save: {exc}")
    """
