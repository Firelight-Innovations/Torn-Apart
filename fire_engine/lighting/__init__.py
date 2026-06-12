"""
fire_engine.lighting — Scene lighting: GPU volumetric cascades + CPU fallback.

Two backends (``config.lighting_backend``; see docs/systems/lighting.md):

GPU ("gpu", default) — camera-centered radiance cascades with ray-marched GI,
voxel-marched sun/moon shadows, dynamic point/area lights, emissive
materials, and froxel volumetric fog/god rays.  Headless halves exported
here (``VolumeWindow``, ``assemble_geometry``, ``MaterialPalette``,
``LightSet``...); the panda3d half lives in ``fire_engine.lighting.gpu``
(``GpuLightingPipeline``) and is deliberately NOT imported by this package.

CPU ("cpu", legacy) — baked-vertex sunlight:
LightGrid
    Per-chunk ``uint8 (16, 16, 16)`` light array store.
occupancy_from_materials
    Downsample a 32³ terrain material array to a 16³ occupancy grid.
SunlightComputer
    Vectorised column-pass + box-blur sunlight; subscribes to terrain events.
make_light_sampler
    Factory returning a ``light_sampler`` callable for ``build_mesh``.

Quick start
-----------
>>> from fire_engine.core import load_config, EventBus
>>> from fire_engine.core.rng import set_world_seed
>>> from fire_engine.terrain import ChunkManager
>>> from fire_engine.lighting import LightGrid, SunlightComputer, make_light_sampler
>>> set_world_seed(1337)
>>> cfg = load_config()
>>> bus = EventBus()
>>> cm = ChunkManager(cfg, bus)
>>> lg = LightGrid()
>>> sc = SunlightComputer(cfg, cm, lg, bus)
>>> sc.recompute_all_loaded()
>>> sampler = make_light_sampler(lg, cfg)
>>> # pass sampler to cm.stream_frame or cm.mesh_chunk
"""

from fire_engine.lighting.light_grid import (
    LightGrid,
    occupancy_from_materials,
    LIGHT_FULL,
    LIGHT_AMBIENT,
)
from fire_engine.lighting.sunlight import (
    SunlightComputer,
    make_light_sampler,
)
from fire_engine.lighting.lights import (
    AreaLight,
    LightSet,
    PointLight,
)
from fire_engine.lighting.palette import (
    MaterialPalette,
    build_default_palette,
)
from fire_engine.lighting.occluders import (
    TreeOccluderSet,
    splat_tree_occluders,
)
from fire_engine.lighting.volume import (
    EMISSION_SCALE,
    ChunkBlockCache,
    GeometryVolume,
    VolumeWindow,
    assemble_geometry,
)

# NOTE: the GPU half (GpuLightingPipeline) lives in fire_engine.lighting.gpu
# and is deliberately NOT imported here — it imports panda3d, and this
# package must stay importable in the headless test suite.  Import it
# explicitly: ``from fire_engine.lighting.gpu import GpuLightingPipeline``.

__all__ = [
    # legacy CPU backend (lighting_backend = "cpu")
    "LightGrid",
    "occupancy_from_materials",
    "LIGHT_FULL",
    "LIGHT_AMBIENT",
    "SunlightComputer",
    "make_light_sampler",
    # GPU volumetric backend — headless halves
    "PointLight",
    "AreaLight",
    "LightSet",
    "MaterialPalette",
    "build_default_palette",
    "VolumeWindow",
    "GeometryVolume",
    "assemble_geometry",
    "ChunkBlockCache",
    "EMISSION_SCALE",
    "TreeOccluderSet",
    "splat_tree_occluders",
]
