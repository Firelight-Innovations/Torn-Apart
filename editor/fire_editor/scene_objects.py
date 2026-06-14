"""Re-export shim — the authoring scene graph now lives in the engine.

The placed-object schema moved to :mod:`fire_engine.scene.objects` so the game
can load editor scenes without the schema existing in two places (DECISIONS.md
2026-06-12; editor imports engine, never the reverse). This shim keeps every
existing ``fire_editor.scene_objects`` import working unchanged.
"""

from fire_engine.scene.objects import (  # noqa: F401
    KINDS,
    SceneError,
    SceneObject,
    SceneObjectStore,
)
