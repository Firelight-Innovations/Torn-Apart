"""Simulation — agent and macro-simulation systems (grouping package).

Holds the engine's Layer-4/5 agency and simulation systems. Most are stubs in
the current milestone (see ``docs/ARCHITECTURE.md`` §5.8–5.11).

Sub-packages
------------
- :mod:`fire_engine.simulation.ai`       — NPC AI (3-tier; stub).
- :mod:`fire_engine.simulation.economy`  — economic simulation (stub).
- :mod:`fire_engine.simulation.politics` — faction / political simulation (stub).
- :mod:`fire_engine.simulation.player`   — player agency (thin; same interface as an NPC agent).

Import sub-package APIs directly, e.g.::

    from fire_engine.simulation.player import FlyController
"""
