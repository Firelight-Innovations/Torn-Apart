"""
tests/render/_impl/test_app_terrain.py — Headless tests for render/_impl/app_terrain.py.

Tests the pure-Python parts of setup_terrain_rendering and
stream_and_upload_terrain using SimpleNamespace fakes.  No panda3d required.
"""

from __future__ import annotations

import types

from fire_engine.render._impl.app_terrain import setup_terrain_rendering, stream_and_upload_terrain


def _make_fake_terrain_root() -> types.SimpleNamespace:
    """A minimal fake for App.terrain_root (records set_texture / set_light_off calls)."""
    root = types.SimpleNamespace()
    root._set_texture_calls: list[object] = []
    root._set_light_off_count: int = 0

    def set_texture(tex: object) -> None:
        root._set_texture_calls.append(tex)

    def set_light_off() -> None:
        root._set_light_off_count += 1

    def attach_new_node(node: object) -> types.SimpleNamespace:
        np_node = types.SimpleNamespace()
        np_node._node = node
        np_node._removed = False

        def remove_node() -> None:
            np_node._removed = True

        np_node.remove_node = remove_node
        return np_node

    root.set_texture = set_texture
    root.set_light_off = set_light_off
    root.attach_new_node = attach_new_node
    return root


def _make_fake_app(chunk_manager: object = None) -> types.SimpleNamespace:
    fake = types.SimpleNamespace()
    fake.terrain_root = _make_fake_terrain_root()
    fake.material_textures = None
    fake.chunk_manager = chunk_manager
    fake._chunk_nodes: dict = {}
    fake.camera_go = types.SimpleNamespace()
    fake.camera_go.transform = types.SimpleNamespace()
    fake.camera_go.transform.position = types.SimpleNamespace(x=0, y=0, z=0)
    fake.light_sampler = None
    return fake


class TestSetupTerrainRendering:
    def test_sets_material_textures(self) -> None:
        fake = _make_fake_app()
        textures = {1: object(), 2: object()}
        setup_terrain_rendering(fake, material_textures=textures)
        assert fake.material_textures is textures

    def test_calls_set_light_off(self) -> None:
        fake = _make_fake_app()
        setup_terrain_rendering(fake)
        assert fake.terrain_root._set_light_off_count == 1

    def test_applies_ground_texture_when_provided(self) -> None:
        fake = _make_fake_app()
        tex = object()
        setup_terrain_rendering(fake, ground_texture=tex)
        assert tex in fake.terrain_root._set_texture_calls

    def test_no_set_texture_when_ground_texture_is_none(self) -> None:
        fake = _make_fake_app()
        setup_terrain_rendering(fake, ground_texture=None)
        assert fake.terrain_root._set_texture_calls == []

    def test_material_textures_none_by_default(self) -> None:
        fake = _make_fake_app()
        setup_terrain_rendering(fake)
        assert fake.material_textures is None


class TestStreamAndUploadTerrain:
    def test_does_nothing_when_chunk_manager_is_none(self) -> None:
        fake = _make_fake_app(chunk_manager=None)
        # Must not raise
        stream_and_upload_terrain(fake)
        assert fake._chunk_nodes == {}

    def test_drains_pending_meshes(self) -> None:
        """Uploaded meshes are removed from pending_meshes and added to _chunk_nodes."""

        class _FakeMesh:
            pass

        class _FakeChunkManager:
            def __init__(self) -> None:
                self.pending_meshes: dict = {(0, 0, 0): _FakeMesh()}
                self.unloaded_this_frame: list = []

            def stream_frame(self, pos: object, light_sampler: object) -> None:
                pass  # no-op for this test

        fake = _make_fake_app(chunk_manager=_FakeChunkManager())

        # Patch geometry_bridge at the module level to avoid panda3d import
        import fire_engine.render.bridges.geometry_bridge as _geo_bridge

        _original = getattr(_geo_bridge, "to_geom_node", None)

        class _FakeGeomNode:
            pass

        def _fake_to_geom_node(mesh: object, name: str, material_textures: object) -> _FakeGeomNode:
            return _FakeGeomNode()

        _geo_bridge.to_geom_node = _fake_to_geom_node  # type: ignore[assignment]
        try:
            stream_and_upload_terrain(fake)
        finally:
            if _original is not None:
                _geo_bridge.to_geom_node = _original  # type: ignore[assignment]

        # pending_meshes should be empty after drain
        assert fake.chunk_manager.pending_meshes == {}
        # The coord should now be tracked in _chunk_nodes
        assert (0, 0, 0) in fake._chunk_nodes

    def test_removes_unloaded_chunks(self) -> None:
        """Coords in unloaded_this_frame are removed from _chunk_nodes."""

        class _FakeChunkManager:
            def __init__(self) -> None:
                self.pending_meshes: dict = {}
                self.unloaded_this_frame: list = [(5, 5, 5)]

            def stream_frame(self, pos: object, light_sampler: object) -> None:
                pass

        fake = _make_fake_app(chunk_manager=_FakeChunkManager())
        # Pre-populate _chunk_nodes with a fake node
        fake_node = types.SimpleNamespace()
        fake_node._removed = False

        def _remove_node() -> None:
            fake_node._removed = True

        fake_node.remove_node = _remove_node
        fake._chunk_nodes[(5, 5, 5)] = fake_node

        stream_and_upload_terrain(fake)

        assert (5, 5, 5) not in fake._chunk_nodes
        assert fake_node._removed is True
