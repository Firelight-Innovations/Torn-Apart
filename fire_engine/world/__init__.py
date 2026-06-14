"""World — the simulated natural environment (grouping package).

Holds the engine's environment-simulation systems. These are all **headless**
(numpy + :mod:`fire_engine.core` only); their Panda3D upload/render bridges live
in :mod:`fire_engine.render`.

Sub-packages
------------
- ``fire_engine.world.terrain`` — voxel terrain: generation, meshing, edits.
- ``fire_engine.world.weather`` — spatial weather sim (cells, fronts, lightning).
- ``fire_engine.world.wind``    — spatially-varying wind field.
- ``fire_engine.world.sky``     — atmosphere, celestial bodies, sky state.

Import sub-package APIs directly, e.g.::

    from fire_engine.world.terrain import ChunkManager

This package intentionally re-exports nothing, keeping import side effects and
cross-package coupling explicit at every call site.
"""
