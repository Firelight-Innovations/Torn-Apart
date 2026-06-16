"""ChunkService — world lifecycle + chunk mesh streaming (EDITOR_PRD Phase E1).

Registers the world/scene RPC methods and streams chunk meshes to the viewport
as binary MESH frames around the editor camera, prioritised nearest-first and
yielding between batches so the control channel stays responsive. Sunlight is
computed (CPU) before meshing so vertex colours match the game.
"""

from __future__ import annotations

import asyncio
import logging

from fire_engine.core.math3d import Vec3
from fire_engine.save import SaveIncompatibleError
from fire_engine.world.terrain import BrushMode

from .._generated import ErrorCode, Method, Notification, SchemaId
from ..binary import encode_frame
from ..commands import EditCommand, SceneCommand, UndoStack
from ..meshcodec import encode_mesh_payload
from ..rpc import RpcError
from ..session import EditorSession
from ..texturecodec import encode_texture_payload

log = logging.getLogger("fire_editor.chunks")

_STREAM_YIELD_EVERY = 8  # mesh this many chunks, then yield to the event loop
_NEIGHBORS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


class ChunkService:
    """Owns world open/save and the chunk-streaming loop for one daemon."""

    def __init__(self, daemon) -> None:
        self.daemon = daemon
        self._payload_seq = 0
        self._client_chunks: set[tuple[int, int, int]] = set()
        self._stream_task: asyncio.Task | None = None
        self.history = UndoStack()
        self._register()

    def _register(self) -> None:
        d = self.daemon.dispatcher
        d.register(Method.WORLD_OPEN, self.world_open)
        d.register(Method.WORLD_SAVE, self.world_save)
        d.register(Method.WORLD_GROUND_LUT, self.ground_lut)
        d.register(Method.CHUNKS_SET_CENTER, self.set_center)
        d.register(Method.SCENE_STATS, self.scene_stats)
        d.register(Method.TERRAIN_RAYCAST, self.raycast)
        d.register(Method.TERRAIN_BRUSH, self.brush)
        d.register(Method.EDIT_UNDO, self.undo)
        d.register(Method.EDIT_REDO, self.redo)

    # ------------------------------------------------------------------ #
    # World lifecycle
    # ------------------------------------------------------------------ #
    async def world_open(self, params: dict) -> dict:
        self._cancel_stream()
        seed = params.get("seed")
        save_path = params.get("save_path")
        if (seed is None) == (save_path is None):
            raise RpcError(ErrorCode.INVALID_PARAMS, "provide exactly one of seed / save_path")
        try:
            if save_path is not None:
                session = EditorSession.from_save(str(save_path))
            else:
                session = EditorSession.from_seed(int(seed))
        except SaveIncompatibleError as e:
            raise RpcError(ErrorCode.APP_ERROR, f"incompatible save: {e}") from e
        except FileNotFoundError as e:
            raise RpcError(ErrorCode.APP_ERROR, f"save not found: {save_path}") from e

        self.daemon.session = session
        self._client_chunks.clear()
        self.history.clear()
        cfg = session.config
        return {
            "ok": True,
            "seed": session.seed,
            "edited_chunks": session.edited_chunk_count(),
            "config": {
                "chunk_size": int(cfg.chunk_size),
                "voxel_size": float(cfg.voxel_size),
                "chunk_meters": float(cfg.chunk_meters),
                "light_grid_scale": int(cfg.light_grid_scale),
                "view_distance_chunks": int(cfg.view_distance_chunks),
                "world_size_m": float(cfg.world_size_m),
                "ground_height_m": float(cfg.ground_height_m),
                "ground_seed": float(session.ground_seed),
                "ground_texels_per_m": float(cfg.ground_texels_per_m),
            },
        }

    async def world_save(self, params: dict) -> dict:
        session = self._require_session()
        path = str(params["path"])
        try:
            session.save(path)
        except OSError as e:
            raise RpcError(ErrorCode.APP_ERROR, f"save failed: {e}") from e
        return {"ok": True, "path": path, "edited_chunks": session.edited_chunk_count()}

    async def ground_lut(self, params: dict) -> dict:
        """Ship the procedural-ground palette LUT as a TEXTURE binary frame."""
        session = self._require_session()
        lut = session.ground_lut()
        self._payload_seq += 1
        pid = self._payload_seq
        await self.daemon.server.broadcast_binary(
            encode_frame(SchemaId.TEXTURE, pid, encode_texture_payload(lut))
        )
        return {
            "ok": True,
            "payload_id": pid,
            "width": int(lut.shape[1]),
            "height": int(lut.shape[0]),
            "ground_seed": float(session.ground_seed),
            "ground_texels_per_m": float(session.config.ground_texels_per_m),
        }

    # ------------------------------------------------------------------ #
    # Streaming
    # ------------------------------------------------------------------ #
    async def set_center(self, params: dict) -> dict:
        session = self._require_session()
        center = Vec3(float(params["x"]), float(params["y"]), float(params["z"]))
        radius = int(params.get("radius") or session.config.view_distance_chunks)
        coords = session.region_coords(center, radius)
        self._cancel_stream()
        if params.get("resend"):
            # A fresh client attached to a running daemon: forget what previous
            # clients were sent so everything in range streams again.
            self._client_chunks.clear()
        self._stream_task = asyncio.create_task(self._stream(session, coords))
        return {"ok": True, "requested": len(coords)}

    async def _stream(self, session: EditorSession, coords: list) -> None:
        try:
            desired = set(coords)
            removed = 0
            for c in list(self._client_chunks):
                if c not in desired:
                    await self.daemon.server.broadcast_notification(
                        Notification.CHUNK_UNLOAD, {"cx": c[0], "cy": c[1], "cz": c[2]}
                    )
                    self._client_chunks.discard(c)
                    removed += 1

            session.ensure_loaded(coords)
            session.relight()

            sent = 0
            for i, coord in enumerate(coords):
                if coord in self._client_chunks:
                    continue
                mesh = session.mesh(coord)
                if mesh.is_empty:
                    continue
                await self._push_mesh(coord, mesh)
                sent += 1
                if i % _STREAM_YIELD_EVERY == 0:
                    await asyncio.sleep(0)

            await self.daemon.server.broadcast_notification(
                Notification.STREAM_DONE, {"sent": sent, "removed": removed}
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("chunk streaming failed")

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    async def scene_stats(self, params: dict) -> dict:
        session = self.daemon.session
        if session is None:
            return {"chunks_loaded": 0, "meshed": 0, "vertices": 0, "triangles": 0}
        meshes = session.cm.pending_meshes.values()
        return {
            "chunks_loaded": len(session.cm.chunks),
            "meshed": len(session.cm.pending_meshes),
            "vertices": sum(m.vertex_count for m in meshes),
            "triangles": sum(m.tri_count for m in meshes),
        }

    async def raycast(self, params: dict) -> dict:
        session = self._require_session()
        origin = Vec3(float(params["ox"]), float(params["oy"]), float(params["oz"]))
        direction = Vec3(float(params["dx"]), float(params["dy"]), float(params["dz"]))
        max_d = float(params.get("max_distance") or 100.0)
        hit = session.raycast(origin, direction, max_d)
        if hit is None:
            return {"hit": None}
        return {
            "hit": {
                "point": [hit.point.x, hit.point.y, hit.point.z],
                "normal": [hit.normal.x, hit.normal.y, hit.normal.z],
                "voxel": list(hit.voxel),
                "chunk": list(hit.chunk_coord),
                "distance": hit.distance,
            }
        }

    # ------------------------------------------------------------------ #
    # Editing (Phase E3)
    # ------------------------------------------------------------------ #
    async def brush(self, params: dict) -> dict:
        session = self._require_session()
        shape = str(params["shape"])
        mode_s = str(params["mode"]).lower()
        if mode_s not in ("add", "remove"):
            raise RpcError(ErrorCode.INVALID_PARAMS, "mode must be 'add' or 'remove'")
        mode = BrushMode.ADD if mode_s == "add" else BrushMode.REMOVE
        material = params.get("material")
        if material is None:
            material = 1 if mode is BrushMode.ADD else 0
        center = Vec3(float(params["x"]), float(params["y"]), float(params["z"]))
        try:
            brush = session.make_brush(
                shape,
                radius=float(params.get("radius") or 2.0),
                hx=float(params.get("hx") or 1.0),
                hy=float(params.get("hy") or 1.0),
                hz=float(params.get("hz") or 1.0),
                height=float(params.get("height") or 2.0),
            )
        except ValueError as e:
            raise RpcError(ErrorCode.INVALID_PARAMS, str(e)) from e

        coords, touched, before, after = session.apply_brush_edit(
            brush, center, mode, int(material)
        )
        self.history.push(EditCommand(f"{shape} {mode_s}", before, after))
        await self._remesh_and_push(session, coords)
        await self._emit_edit_state(session)
        return {
            "ok": True,
            "touched": len(touched),
            "can_undo": self.history.can_undo,
            "can_redo": self.history.can_redo,
        }

    async def undo(self, params: dict) -> dict:
        session = self._require_session()
        cmd = self.history.undo()
        if cmd is None:
            return {
                "ok": False,
                "touched": 0,
                "label": "",
                "can_undo": False,
                "can_redo": self.history.can_redo,
            }
        if isinstance(cmd, SceneCommand):
            return await self._apply_scene_command(session, cmd, cmd.before_delta)
        session.restore(cmd.before)
        await self._remesh_and_push(session, cmd.coords)
        await self._emit_edit_state(session)
        return {
            "ok": True,
            "touched": len(cmd.coords),
            "label": cmd.label,
            "can_undo": self.history.can_undo,
            "can_redo": self.history.can_redo,
        }

    async def redo(self, params: dict) -> dict:
        session = self._require_session()
        cmd = self.history.redo()
        if cmd is None:
            return {
                "ok": False,
                "touched": 0,
                "label": "",
                "can_undo": self.history.can_undo,
                "can_redo": False,
            }
        if isinstance(cmd, SceneCommand):
            return await self._apply_scene_command(session, cmd, cmd.after_delta)
        session.restore(cmd.after)
        await self._remesh_and_push(session, cmd.coords)
        await self._emit_edit_state(session)
        return {
            "ok": True,
            "touched": len(cmd.coords),
            "label": cmd.label,
            "can_undo": self.history.can_undo,
            "can_redo": self.history.can_redo,
        }

    async def _apply_scene_command(
        self, session: EditorSession, cmd: SceneCommand, delta: dict
    ) -> dict:
        """Restore a scene-hierarchy snapshot (undo/redo of a scene op)."""
        session.scene.apply_delta(delta)
        await self.daemon.server.broadcast_notification(
            Notification.SCENE_CHANGED, {"objects": session.scene.tree()}
        )
        await self._emit_edit_state(session)
        return {
            "ok": True,
            "touched": 0,
            "label": cmd.label,
            "can_undo": self.history.can_undo,
            "can_redo": self.history.can_redo,
        }

    # ------------------------------------------------------------------ #
    # Push helpers
    # ------------------------------------------------------------------ #
    async def _push_mesh(self, coord: tuple[int, int, int], mesh) -> None:
        """Announce + send one chunk's MESH frame, recording it as client-visible."""
        self._payload_seq += 1
        pid = self._payload_seq
        await self.daemon.server.broadcast_notification(
            Notification.CHUNK_READY,
            {
                "cx": coord[0],
                "cy": coord[1],
                "cz": coord[2],
                "payload_id": pid,
                "vertices": mesh.vertex_count,
                "triangles": mesh.tri_count,
            },
        )
        await self.daemon.server.broadcast_binary(
            encode_frame(SchemaId.MESH, pid, encode_mesh_payload(coord, mesh))
        )
        self._client_chunks.add(coord)

    async def _push_unload(self, coord: tuple[int, int, int]) -> None:
        await self.daemon.server.broadcast_notification(
            Notification.CHUNK_UNLOAD, {"cx": coord[0], "cy": coord[1], "cz": coord[2]}
        )
        self._client_chunks.discard(coord)

    async def _remesh_and_push(self, session: EditorSession, coords) -> int:
        """Relight + remesh edited chunks (and loaded neighbours) and stream them.

        A neighbour must remesh too: removing a boundary voxel exposes a face on
        the adjacent chunk. Chunks that became empty are unloaded on the client.
        """
        affected: set[tuple[int, int, int]] = set()
        for c in coords:
            affected.add(c)
            for d in _NEIGHBORS:
                n = (c[0] + d[0], c[1] + d[1], c[2] + d[2])
                if n in session.cm.chunks:
                    affected.add(n)
        session.relight()
        sent = 0
        for coord in affected:
            if coord not in session.cm.chunks:
                continue
            mesh = session.mesh(coord)
            if mesh.is_empty:
                if coord in self._client_chunks:
                    await self._push_unload(coord)
                continue
            await self._push_mesh(coord, mesh)
            sent += 1
        return sent

    async def _emit_edit_state(self, session: EditorSession) -> None:
        await self.daemon.server.broadcast_notification(
            Notification.EDIT_STATE,
            {
                "can_undo": self.history.can_undo,
                "can_redo": self.history.can_redo,
                "edited_chunks": session.edited_chunk_count(),
            },
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _require_session(self) -> EditorSession:
        s = self.daemon.session
        if s is None:
            raise RpcError(ErrorCode.APP_ERROR, "no world open; call world.open first")
        return s

    def _cancel_stream(self) -> None:
        if self._stream_task is not None and not self._stream_task.done():
            self._stream_task.cancel()
        self._stream_task = None
