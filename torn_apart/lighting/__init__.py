"""
torn_apart.lighting — Scene lighting: GPU volumetric cascades + CPU fallback.

Two backends (``config.lighting_backend``; see docs/systems/lighting.md):

GPU ("gpu", default) — camera-centered radiance cascades with flood-fill GI,
voxel-marched sun/moon shadows, dynamic point/area lights, emissive
materials, and froxel volumetric fog/god rays.  Headless halves exported
here (``VolumeWindow``, ``assemble_geometry``, ``MaterialPalette``,
``LightSet``...); the panda3d half lives in ``torn_apart.lighting.gpu``
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
>>> from torn_apart.core import load_config, EventBus
>>> from torn_apart.core.rng import set_world_seed
>>> from torn_apart.terrain import ChunkManager
>>> from torn_apart.lighting import LightGrid, SunlightComputer, make_light_sampler
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

from torn_apart.lighting.light_grid import (
    LightGrid,
    occupancy_from_materials,
    LIGHT_FULL,
    LIGHT_AMBIENT,
)
from torn_apart.lighting.sunlight import (
    SunlightComputer,
    make_light_sampler,
)
from torn_apart.lighting.lights import (
    AreaLight,
    LightSet,
    PointLight,
)
from torn_apart.lighting.palette import (
    MaterialPalette,
    build_default_palette,
)
from torn_apart.lighting.volume import (
    EMISSION_SCALE,
    GeometryVolume,
    VolumeWindow,
    assemble_geometry,
)

# NOTE: the GPU half (GpuLightingPipeline) lives in torn_apart.lighting.gpu
# and is deliberately NOT imported here — it imports panda3d, and this
# package must stay importable in the headless test suite.  Import it
# explicitly: ``from torn_apart.lighting.gpu import GpuLightingPipeline``.

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
    "EMISSION_SCALE",
]
