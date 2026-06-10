"""ChunkService — world lifecycle + chunk mesh streaming (EDITOR_PRD Phase E1).

Registers the world/scene RPC methods and streams chunk meshes to the viewport
as binary MESH frames around the editor camera, prioritised nearest-first and
yielding between batches so the control channel stays responsive. Sunlight is
computed (CPU) before meshing so vertex colours match the game.
"""
from __future__ import annotations

import asyncio
import logging

from torn_apart.core.math3d import Vec3
from torn_apart.save import SaveIncompatibleError

from .._generated import ErrorCode, Method, Notification, SchemaId
from ..binary import encode_frame
from ..meshcodec import encode_mesh_payload
from ..rpc import RpcError
from ..session import EditorSession

log = logging.getLogger("fire_editor.chunks")

_STREAM_YIELD_EVERY = 8  # mesh this many chunks, then yield to the event loop


class ChunkService:
    """Owns world open/save and the chunk-streaming loop for one daemon."""

    def __init__(self, daemon) -> None:
        self.daemon = daemon
        self._payload_seq = 0
        self._client_chunks: set[tuple[int, int, int]] = set()
        self._stream_task: asyncio.Task | None = None
        self._register()

    def _register(self) -> None:
        d = self.daemon.dispatcher
        d.register(Method.WORLD_OPEN, self.world_open)
        d.register(Method.WORLD_SAVE, self.world_save)
        d.register(Method.CHUNKS_SET_CENTER, self.set_center)
        d.register(Method.SCENE_STATS, self.scene_stats)
        d.register(Method.TERRAIN_RAYCAST, self.raycast)

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
            raise RpcError(ErrorCode.APP_ERROR, f"incompatible save: {e}")
        except FileNotFoundError:
            raise RpcError(ErrorCode.APP_ERROR, f"save not found: {save_path}")

        self.daemon.session = session
        self._client_chunks.clear()
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
            },
        }

    async def world_save(self, params: dict) -> dict:
        session = self._require_session()
        path = str(params["path"])
        try:
            session.save(path)
        except OSError as e:
            raise RpcError(ErrorCode.APP_ERROR, f"save failed: {e}")
        return {"ok": True, "path": path, "edited_chunks": session.edited_chunk_count()}

    # ------------------------------------------------------------------ #
    # Streaming
    # ------------------------------------------------------------------ #
    async def set_center(self, params: dict) -> dict:
        session = self._require_session()
        center = Vec3(float(params["x"]), float(params["y"]), float(params["z"]))
        radius = int(params.get("radius") or session.config.view_distance_chunks)
        coords = session.region_coords(center, radius)
        self._cancel_stream()
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
                self._payload_seq += 1
                pid = self._payload_seq
                await self.daemon.server.broadcast_notification(
                    Notification.CHUNK_READY,
                    {
                        "cx": coord[0], "cy": coord[1], "cz": coord[2],
                        "payload_id": pid,
                        "vertices": mesh.vertex_count, "triangles": mesh.tri_count,
                    },
                )
                await self.daemon.server.broadcast_binary(
                    encode_frame(SchemaId.MESH, pid, encode_mesh_payload(coord, mesh))
                )
                self._client_chunks.add(coord)
                sent += 1
                if i % _STREAM_YIELD_EVERY == 0:
                    await asyncio.sleep(0)

            await self.daemon.server.broadcast_notification(
                Notification.STREAM_DONE, {"sent": sent, "removed": removed}
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — background task: log, don't crash the daemon
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
