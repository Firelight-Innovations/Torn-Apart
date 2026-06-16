"""
render/profiler_bridge.py — mirror the core profiler into Panda3D PStats.

The core :class:`~fire_engine.core.profiler.Profiler` is panda3d-free; this
bridge subscribes to its observer hook and, for each named scope, lazily
creates a matching ``PStatCollector`` and calls ``.start()`` / ``.stop()`` in
lockstep (a 1:1 mapping — the core profiler invokes the observer only on the
outermost enter/exit of each scope, exactly mirroring PStats semantics).
Counters are mirrored via ``set_level``.

This gives the **human** Panda3D's full PStats GUI for free — timeline, flame
graph, and the built-in App / Cull / Draw / Flip split — alongside our custom
collectors (``Update:Weather``, ``ChunkStream``, …).  The in-engine overlay does
NOT depend on this; PStats only displays anything when the standalone ``pstats``
GUI is running and this client has connected.

Panda3D imports are allowed here per ARCHITECTURE.md §3 (this module lives in
``world/``).

How the human reads it
----------------------
1. Launch the GUI server (ships with Panda3D)::

       pstats                       # Windows: pstats.exe, listens on :5185

2. Run the game with ``profiler_pstats = true`` in ``[profiler]`` (or
   ``tools/profile_run.py --pstats``).  This client connects on boot.
3. In the GUI, open the **Flame Graph** view to see the per-frame stage
   breakdown; our scopes appear as named bars next to App/Cull/Draw.

Example
-------
    from fire_engine.core.profiler import get_profiler
    from fire_engine.render.bridges.profiler_bridge import PStatsBridge

    bridge = PStatsBridge(get_profiler(), connect=True)
    # ... run frames; the pstats GUI now shows Update:Weather etc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Panda3D imports allowed in world/ per ARCHITECTURE §3.
from panda3d.core import PStatClient, PStatCollector

from fire_engine.core.log import get_logger

if TYPE_CHECKING:
    from fire_engine.core.profiler import Profiler

_log = get_logger("profiler")


class PStatsBridge:
    """
    Observer that mirrors core profiler scopes/counters into PStatCollectors.

    Parameters
    ----------
    profiler : Profiler
        The core profiler to mirror.  The bridge registers itself as an
        observer; nothing happens until scopes run.
    connect : bool
        When True, connect this process to a PStats server immediately (so the
        ``pstats`` GUI can attach).  Connection failure is logged, not fatal.

    Notes
    -----
    - ``PStatCollector`` accepts a compound, colon-separated name (the same
      convention our scope names use), so ``"Update:Weather"`` nests under
      ``Update`` in the GUI automatically.
    - Collectors are cached by name; creating one per unique scope is cheap and
      one-time.
    """

    def __init__(self, profiler: Profiler, connect: bool = False) -> None:
        self._timers: dict[str, PStatCollector] = {}
        self._counters: dict[str, PStatCollector] = {}
        profiler.add_observer(self._on_start, self._on_stop)
        profiler.add_counter_observer(self._on_counter)
        if connect:
            self.connect()

    @staticmethod
    def connect(hostname: str = "", port: int = -1) -> bool:
        """
        Connect this process to a PStats server.

        Parameters
        ----------
        hostname : str — server host ("" = localhost / the PRC default).
        port : int — server port (-1 = the PRC default, 5185).

        Returns
        -------
        bool — True if connected (or already connected).
        """
        try:
            ok = PStatClient.connect(hostname, port)
            if ok:
                _log.info("PStats connected (run the 'pstats' GUI to view)")
            else:
                _log.warning(
                    "PStats connect() returned False — is the 'pstats' GUI "
                    "server running? (overlay still works without it)"
                )
            return bool(ok)
        except Exception as exc:
            _log.warning("PStats connect failed: %s", exc)
            return False

    # -- observer callbacks (called from core.profiler, panda3d-free side) --

    def _timer(self, name: str) -> PStatCollector:
        c = self._timers.get(name)
        if c is None:
            c = self._timers[name] = PStatCollector(name)
        return c

    def _on_start(self, name: str) -> None:
        self._timer(name).start()

    def _on_stop(self, name: str) -> None:
        self._timer(name).stop()

    def _on_counter(self, name: str, value: float) -> None:
        c = self._counters.get(name)
        if c is None:
            c = self._counters[name] = PStatCollector(name)
        c.set_level(float(value))
