"""``fire_editor`` — the Fire Editor daemon (EDITOR_PRD).

Headless host that runs the Torn Apart engine with the game closed and serves a
WebSocket protocol to the VS Code / Cursor extension. Imports ``torn_apart``
public APIs only; **never imports panda3d** (hard rule 1).

Public surface (Phase E0):

- :class:`~fire_editor.daemon.Daemon` — dispatcher + server + handshake.
- :func:`~fire_editor.binary.encode_frame` / :func:`~fire_editor.binary.decode_frame`.
- :class:`~fire_editor.rpc.Dispatcher`, :class:`~fire_editor.rpc.RpcError`.
- Constants in :mod:`fire_editor._generated` (codegen from ``protocol/schema.json``).
"""
from __future__ import annotations

from ._generated import DAEMON_VERSION, PROTOCOL_VERSION
from .binary import BinaryFrameError, decode_frame, encode_frame
from .daemon import Daemon
from .meshcodec import decode_mesh_payload, encode_mesh_payload
from .rpc import Dispatcher, RpcError
from .session import EditorSession

__all__ = [
    "Daemon",
    "Dispatcher",
    "RpcError",
    "EditorSession",
    "encode_frame",
    "decode_frame",
    "BinaryFrameError",
    "encode_mesh_payload",
    "decode_mesh_payload",
    "PROTOCOL_VERSION",
    "DAEMON_VERSION",
]
