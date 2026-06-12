"""
fire_engine.scene — authored-scene data model and game-side runtime loader.

Headless package (zero panda3d imports). Holds the placed-object schema the
Fire Editor writes into ``.ta`` saves (save_key ``"editor_scene"``) and the
:class:`SceneRuntime` Saveable the game registers to instantiate those objects
as live GameObjects on load. Visuals (cube/sphere models, point lights) are
delegated to a factory constructed in ``world/`` (``SceneVisualFactory``) so
this package stays headless-testable.
"""

from fire_engine.scene.objects import (
    KINDS,
    SceneError,
    SceneObject,
    SceneObjectStore,
)
from fire_engine.scene.runtime import SceneRuntime

__all__ = [
    "KINDS",
    "SceneError",
    "SceneObject",
    "SceneObjectStore",
    "SceneRuntime",
]
