"""
tests/world/terrain/lod/test_types.py — LodJob / LodResult dataclass contracts.

Headless: no panda3d imports.  Confirms the hand-off types are frozen
(immutable snapshots, Hard Rule 12) and round-trip their fields verbatim.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from fire_engine.world.terrain.lod.types import LodJob, LodResult
from fire_engine.world.terrain.meshing import MeshArrays


def _empty_mesh() -> MeshArrays:
    return MeshArrays(
        positions=np.zeros((0, 3), np.float32),
        normals=np.zeros((0, 3), np.float32),
        uvs=np.zeros((0, 2), np.float32),
        colors=np.zeros((0, 4), np.float32),
        indices=np.zeros((0,), np.uint32),
    )


class TestLodJob:
    def test_fields_round_trip(self):
        mats = np.zeros((32, 32, 32), dtype=np.uint8)
        nb = {(1, 0, 0): np.ones((32, 32, 32), dtype=np.uint8)}
        job = LodJob(
            coord=(1, 2, 3),
            materials=mats,
            neighbors=nb,
            chunk_size=32,
            voxel_size=0.5,
            shade_strength=0.25,
            mesh_style="faceted",
            seq=9,
        )
        assert job.coord == (1, 2, 3)
        assert job.materials is mats
        assert job.neighbors is nb
        assert job.chunk_size == 32
        assert job.voxel_size == 0.5
        assert job.shade_strength == 0.25
        assert job.mesh_style == "faceted"
        assert job.seq == 9

    def test_is_frozen(self):
        job = LodJob((0, 0, 0), np.zeros((1,), np.uint8), {}, 32, 0.5, 0.25, "faceted", 0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            job.seq = 1  # type: ignore[misc]


class TestLodResult:
    def test_fields_round_trip(self):
        mesh = _empty_mesh()
        res = LodResult(coord=(4, 5, 6), mesh=mesh, seq=11)
        assert res.coord == (4, 5, 6)
        assert res.mesh is mesh
        assert res.seq == 11

    def test_is_frozen(self):
        res = LodResult((0, 0, 0), _empty_mesh(), 0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            res.seq = 2  # type: ignore[misc]
