"""
tests/render/_impl/test_app_terrain.py — Headless tests for render/_impl/app_terrain.py.

Tests the pure-Python parts of setup_terrain_rendering,
stream_and_upload_terrain, and _upload_coarse_terrain using SimpleNamespace
fakes.  No panda3d required — the existing fake harness drives the coarse LOD
upload path (P2) headlessly.
"""

from __future__ import annotations

import types
from typing import Any

from fire_engine.render._impl.app_terrain import (
    _upload_coarse_terrain,
    setup_terrain_rendering,
    stream_and_upload_terrain,
)


def _make_fake_node() -> types.SimpleNamespace:
    """A minimal fake NodePath that records remove_node()."""
    node = types.SimpleNamespace()
    node._removed = False

    def remove_node() -> None:
        node._removed = True

    node.remove_node = remove_node
    return node


def _make_fake_terrain_root() -> types.SimpleNamespace:
    """A minimal fake for App.terrain_root (records set_texture / set_light_off /
    attach_new_node calls)."""
    root = types.SimpleNamespace()
    root._set_texture_calls: list[object] = []
    root._set_light_off_count: int = 0
    root._attached: list[object] = []

    def set_texture(tex: object) -> None:
        root._set_texture_calls.append(tex)

    def set_light_off() -> None:
        root._set_light_off_count += 1

    def attach_new_node(geom_node: object) -> types.SimpleNamespace:
        np_node = _make_fake_node()
        np_node._node = geom_node
        root._attached.append(np_node)
        return np_node

    root.set_texture = set_texture
    root.set_light_off = set_light_off
    root.attach_new_node = attach_new_node
    return root


def _make_fake_config(max_uploads: int = 8, coarse_uploads: int = 4) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        lod_max_uploads_per_frame=max_uploads,
        lod_coarse_uploads_per_frame=coarse_uploads,
    )


def _make_fake_app(chunk_manager: object = None) -> types.SimpleNamespace:
    fake = types.SimpleNamespace()
    fake.terrain_root = _make_fake_terrain_root()
    fake.material_textures = None
    fake.chunk_manager = chunk_manager
    fake._chunk_nodes: dict = {}
    fake._coarse_nodes: dict = {}
    fake.camera_go = types.SimpleNamespace()
    fake.camera_go.transform = types.SimpleNamespace()
    fake.camera_go.transform.position = types.SimpleNamespace(x=0, y=0, z=0)
    fake.light_sampler = None
    return fake


class _FakeMesh:
    """A fake MeshArrays with the only field the upload path reads (is_empty)."""

    def __init__(self, is_empty: bool = False) -> None:
        self.is_empty = is_empty


def _install_fake_geom_node() -> tuple[Any, Any]:
    """Patch geometry_bridge.to_geom_node so the upload path runs without panda3d.

    Returns ``(restore_callable, calls_list)`` — call ``restore_callable()`` in a
    finally to put the original back; ``calls_list`` records each (mesh, name).
    """
    import fire_engine.render.bridges.geometry_bridge as _geo_bridge

    original = getattr(_geo_bridge, "to_geom_node", None)
    calls: list[tuple[object, str]] = []

    class _FakeGeomNode:
        def __init__(self, name: str) -> None:
            self.name = name

    def _fake_to_geom_node(mesh: object, name: str, material_textures: object) -> _FakeGeomNode:
        calls.append((mesh, name))
        return _FakeGeomNode(name)

    _geo_bridge.to_geom_node = _fake_to_geom_node  # type: ignore[assignment]

    def _restore() -> None:
        if original is not None:
            _geo_bridge.to_geom_node = original  # type: ignore[assignment]

    return _restore, calls


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


class _FakeChunkManager:
    """Fake ChunkManager for the near-stream path: minimal camera_chunk + config."""

    def __init__(
        self,
        pending: dict | None = None,
        unloaded: list | None = None,
        config: object | None = None,
    ) -> None:
        self.pending_meshes: dict = pending if pending is not None else {}
        self.unloaded_this_frame: list = unloaded if unloaded is not None else []
        self.config = config if config is not None else _make_fake_config()

    def stream_frame(self, pos: object, light_sampler: object) -> None:
        pass  # no-op for these tests

    def camera_chunk(self, pos: object) -> tuple[int, int, int]:
        return (0, 0, 0)


class TestStreamAndUploadTerrain:
    def test_does_nothing_when_chunk_manager_is_none(self) -> None:
        fake = _make_fake_app(chunk_manager=None)
        # Must not raise
        stream_and_upload_terrain(fake)
        assert fake._chunk_nodes == {}

    def test_drains_pending_meshes(self) -> None:
        """Uploaded meshes are removed from pending_meshes and added to _chunk_nodes."""
        cm = _FakeChunkManager(pending={(0, 0, 0): _FakeMesh()})
        fake = _make_fake_app(chunk_manager=cm)

        restore, _calls = _install_fake_geom_node()
        try:
            stream_and_upload_terrain(fake)
        finally:
            restore()

        assert fake.chunk_manager.pending_meshes == {}
        assert (0, 0, 0) in fake._chunk_nodes

    def test_removes_unloaded_chunks(self) -> None:
        """Coords in unloaded_this_frame are removed from _chunk_nodes."""
        cm = _FakeChunkManager(unloaded=[(5, 5, 5)])
        fake = _make_fake_app(chunk_manager=cm)
        fake_node = _make_fake_node()
        fake._chunk_nodes[(5, 5, 5)] = fake_node

        restore, _calls = _install_fake_geom_node()
        try:
            stream_and_upload_terrain(fake)
        finally:
            restore()

        assert (5, 5, 5) not in fake._chunk_nodes
        assert fake_node._removed is True

    def test_no_coarse_attr_is_clean_noop(self) -> None:
        """A chunk_manager without pending_coarse_meshes leaves _coarse_nodes empty."""
        cm = _FakeChunkManager(pending={(0, 0, 0): _FakeMesh()})
        # No coarse channels on this manager (no coarse_streamer wired).
        fake = _make_fake_app(chunk_manager=cm)

        restore, _calls = _install_fake_geom_node()
        try:
            stream_and_upload_terrain(fake)
        finally:
            restore()

        assert fake._coarse_nodes == {}

    def test_end_to_end_coarse_geoms_land_under_terrain_root(self) -> None:
        """A full frame with a coarse_streamer populating the channels uploads
        coarse Geoms, and a near coord and a coarse key are never both present
        (the hard band cut)."""

        class _FakeCoarseStreamer:
            """Populates the coarse channels exactly like CoarseLodStreamer would."""

            def __init__(self, cm: Any) -> None:
                self._cm = cm

            def stream_frame(self, pos: object) -> None:
                # rank-1 node far from the camera; a near chunk near the origin.
                self._cm.pending_coarse_meshes[(1, 4, 0, 0)] = _FakeMesh()

        cm = _FakeChunkManager(pending={(0, 0, 0): _FakeMesh()})
        cm.pending_coarse_meshes = {}
        cm.unloaded_coarse_this_frame = []
        fake = _make_fake_app(chunk_manager=cm)
        fake.coarse_streamer = _FakeCoarseStreamer(cm)

        restore, _calls = _install_fake_geom_node()
        try:
            stream_and_upload_terrain(fake)
        finally:
            restore()

        # Both the near coord and the coarse node uploaded and are tracked.
        assert (0, 0, 0) in fake._chunk_nodes
        assert (1, 4, 0, 0) in fake._coarse_nodes
        # Geoms landed under terrain_root (near + coarse both attached).
        assert len(fake.terrain_root._attached) == 2
        # Hard band cut: the coarse node's covered L0 columns must not also be
        # near chunks (a near coord and a coarse key never both present).
        from fire_engine.world.terrain.lod.node import LodNode

        coarse_cols = {(c[0], c[1]) for c in LodNode(1, 4, 0, 0).covered_chunks()}
        near_cols = {(x, y) for (x, y, _z) in fake._chunk_nodes}
        assert near_cols.isdisjoint(coarse_cols)


class TestUploadCoarseTerrain:
    """Direct coverage of _upload_coarse_terrain via the fake harness (P2)."""

    def _run(self, fake: Any, camera_chunk: tuple[int, int, int] = (0, 0, 0)) -> Any:
        restore, calls = _install_fake_geom_node()
        try:
            from fire_engine.render.bridges.geometry_bridge import to_geom_node

            _upload_coarse_terrain(fake, to_geom_node, camera_chunk)
        finally:
            restore()
        return calls

    def test_noop_when_no_pending_coarse_attr(self) -> None:
        cm = _FakeChunkManager()  # no pending_coarse_meshes attr
        fake = _make_fake_app(chunk_manager=cm)
        calls = self._run(fake)
        assert calls == []
        assert fake._coarse_nodes == {}

    def test_budget_caps_uploads_nearest_first(self) -> None:
        cm = _FakeChunkManager(config=_make_fake_config(coarse_uploads=2))
        # Three rank-1 nodes at increasing distance from camera chunk (0,0,0).
        cm.pending_coarse_meshes = {
            (1, 1, 0, 0): _FakeMesh(),  # nearest
            (1, 5, 0, 0): _FakeMesh(),  # mid
            (1, 9, 0, 0): _FakeMesh(),  # far
        }
        cm.unloaded_coarse_this_frame = []
        fake = _make_fake_app(chunk_manager=cm)

        calls = self._run(fake)

        # At most lod_coarse_uploads_per_frame (2) attach this frame.
        assert len(calls) == 2
        # Nearest-first: the two nearest keys uploaded; the far one stays pending.
        assert (1, 1, 0, 0) in fake._coarse_nodes
        assert (1, 5, 0, 0) in fake._coarse_nodes
        assert (1, 9, 0, 0) not in fake._coarse_nodes
        assert (1, 9, 0, 0) in cm.pending_coarse_meshes

    def test_empty_coarse_mesh_attaches_nothing_but_is_popped(self) -> None:
        cm = _FakeChunkManager(config=_make_fake_config(coarse_uploads=4))
        cm.pending_coarse_meshes = {(1, 1, 0, 0): _FakeMesh(is_empty=True)}
        cm.unloaded_coarse_this_frame = []
        fake = _make_fake_app(chunk_manager=cm)

        calls = self._run(fake)

        # All-air coarse node: no Geom built, no node tracked, but popped.
        assert calls == []
        assert (1, 1, 0, 0) not in fake._coarse_nodes
        assert cm.pending_coarse_meshes == {}
        assert fake.terrain_root._attached == []

    def test_remesh_detaches_old_nodepath_first(self) -> None:
        cm = _FakeChunkManager(config=_make_fake_config(coarse_uploads=4))
        cm.pending_coarse_meshes = {(1, 1, 0, 0): _FakeMesh()}
        cm.unloaded_coarse_this_frame = []
        fake = _make_fake_app(chunk_manager=cm)
        # Pre-existing stale NodePath for the same coarse key (an earlier mesh).
        stale = _make_fake_node()
        fake._coarse_nodes[(1, 1, 0, 0)] = stale

        self._run(fake)

        # The old NodePath was detached and replaced by the new upload.
        assert stale._removed is True
        assert (1, 1, 0, 0) in fake._coarse_nodes
        assert fake._coarse_nodes[(1, 1, 0, 0)] is not stale

    def test_unloaded_coarse_keys_detached_and_forgotten(self) -> None:
        cm = _FakeChunkManager(config=_make_fake_config(coarse_uploads=4))
        cm.pending_coarse_meshes = {}
        cm.unloaded_coarse_this_frame = [(2, 0, 0, 0), (1, 7, 0, 0)]
        fake = _make_fake_app(chunk_manager=cm)
        n1 = _make_fake_node()
        n2 = _make_fake_node()
        fake._coarse_nodes[(2, 0, 0, 0)] = n1
        fake._coarse_nodes[(1, 7, 0, 0)] = n2

        self._run(fake)

        # Keys in unloaded_coarse_this_frame are detached + forgotten (band cut).
        assert (2, 0, 0, 0) not in fake._coarse_nodes
        assert (1, 7, 0, 0) not in fake._coarse_nodes
        assert n1._removed is True
        assert n2._removed is True

    def test_unloaded_coarse_key_with_no_node_is_safe(self) -> None:
        cm = _FakeChunkManager(config=_make_fake_config(coarse_uploads=4))
        cm.pending_coarse_meshes = {}
        cm.unloaded_coarse_this_frame = [(3, 0, 0, 0)]  # never had a node
        fake = _make_fake_app(chunk_manager=cm)

        # Must not raise when retiring a key that was never uploaded.
        self._run(fake)
        assert fake._coarse_nodes == {}
