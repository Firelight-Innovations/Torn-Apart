"""
fire_engine.core._impl — private implementation helpers for core/.

This sub-package holds modules that were split out of their parent modules to
satisfy the one-public-class and <500-lines rules, but cannot live directly in
``fire_engine/core/`` without pushing that package past the 10-module limit.

**Do not import from this package directly in production code.**  All public
names are re-exported from the appropriate top-level core module:

  - ``Quat``                      → ``fire_engine.core.math3d``
  - ``GRAPHICS_PRESETS``,
    ``load_config``,
    ``resolve_graphics_preset``   → ``fire_engine.core.config``
  - ``NullScope``, ``frame_time_stats``
    (and profiler internals)      → ``fire_engine.core.profiler``

Docs: docs/systems/core.md
"""
