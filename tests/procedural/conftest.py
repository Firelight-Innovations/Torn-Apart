"""
tests/procedural/conftest.py — shared fixtures for the procedural test package.
"""

from __future__ import annotations


def restore_builtins() -> None:
    """
    Re-register all built-in ProceduralDef instances.

    Call in teardown_method of any test class that uses reset_registry(),
    so that downstream tests can still access the auto-registered defs.
    """
    from fire_engine.procedural.flora.species.berry_bush import BerryBushDef
    from fire_engine.procedural.flora.species.dead_tree import DeadTreeDef
    from fire_engine.procedural.flora.species.gnarled_oak import GnarledOakDef
    from fire_engine.procedural.flora.species.scrub_bush import ScrubBushDef
    from fire_engine.procedural.registry import register
    from fire_engine.procedural.textures.ground.dirt_ground import DirtGroundDef
    from fire_engine.procedural.textures.ground.grass_ground import GrassGroundDef
    from fire_engine.procedural.textures.ground.wasteland_ground import WastelandGroundDef
    from fire_engine.procedural.textures.sky.moon_surface import MoonSurfaceDef
    from fire_engine.procedural.textures.sky.night_sky import NightSkyDef
    from fire_engine.procedural.textures.sky.night_sky_cube import NightSkyCubeDef
    from fire_engine.procedural.textures.sky.rain_streak import RainStreakDef
    from fire_engine.procedural.textures.sprites.dust_mote import DustMoteDef
    from fire_engine.procedural.textures.sprites.flower_sprite import FlowerSpriteDef
    from fire_engine.procedural.textures.sprites.grass_tuft import GrassTuftDef
    from fire_engine.procedural.textures.sprites.leaf_sprite import LeafSpriteDef

    for cls in (
        WastelandGroundDef,
        GrassGroundDef,
        DirtGroundDef,
        NightSkyDef,
        NightSkyCubeDef,
        RainStreakDef,
        MoonSurfaceDef,
        GrassTuftDef,
        DustMoteDef,
        FlowerSpriteDef,
        LeafSpriteDef,
        BerryBushDef,
        DeadTreeDef,
        GnarledOakDef,
        ScrubBushDef,
    ):
        register(cls())
