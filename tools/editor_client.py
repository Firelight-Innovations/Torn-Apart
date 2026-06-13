"""tools/editor_client.py — drive the Fire Editor daemon from the command line.

This is the agent-facing CLI half of the editor harness (EDITOR_PRD agent
access): every subcommand maps ~1:1 to a daemon RPC method, so an agent (or a
human, or CI) can open a world, stream/inspect chunks, carve terrain, place and
transform scene objects, undo/redo and save — all without the VS Code UI.

Two ways to run it:

  • Persistent server (recommended for an interactive session)::

        python tools/editor_client.py serve --port 8123 --http-port 8770
        # then, in another shell, talk to that same daemon:
        python tools/editor_client.py --port 8123 open --seed 1337
        python tools/editor_client.py --port 8123 brush --x 0 --y 0 --z 7.5 --mode remove
        python tools/editor_client.py --port 8123 create cube --x 2 --y 0 --z 8

    ``serve`` also hosts the browser viewport harness over HTTP and prints its
    URL; the CLI edits above render live in that page (the daemon broadcasts to
    every connected client). Point Chrome (or the Chrome MCP tools) at the URL
    to *see* what your CLI edits do.

  • One-shot (spawns a throwaway daemon, runs one command, exits): omit
    ``--port`` and the CLI spawns its own daemon. Handy for ``tree``/``raycast``
    smoke checks; useless for stateful sequences (each invocation is a fresh
    world).

Add ``--json`` for machine-readable output (the raw RPC result as JSON).

No panda3d import here (hard rule 1) — pure protocol client over fire_editor.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import http.server
import json
import socketserver
import sys
import threading
from pathlib import Path

# Make `fire_engine` (repo root) and `fire_editor` (editor/) importable when run
# as `python tools/editor_client.py`, mirroring the extension's PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_EDITOR_DIR = _REPO_ROOT / "editor"
for _p in (_REPO_ROOT, _EDITOR_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from fire_editor import Daemon, EditorClient, RpcRemoteError, spawn_daemon  # noqa: E402

_DEFAULT_PORT = 8123
_DEFAULT_HTTP_PORT = 8770


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def _emit(result: object, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result), flush=True)
    else:
        print(json.dumps(result, indent=2, default=str), flush=True)


# --------------------------------------------------------------------------- #
# Subcommand bodies — each receives (client, args) and returns the RPC result.
# --------------------------------------------------------------------------- #
async def _cmd_open(c: EditorClient, args) -> object:
    if (args.seed is None) == (args.save is None):
        raise SystemExit("open: provide exactly one of --seed / --save")
    params = {"seed": args.seed} if args.seed is not None else {"save_path": args.save}
    return await c.request("world.open", params)


async def _cmd_save(c: EditorClient, args) -> object:
    return await c.request("world.save", {"path": args.path})


async def _cmd_ground_lut(c: EditorClient, args) -> object:
    return await c.request("world.ground_lut", {})


async def _cmd_set_center(c: EditorClient, args) -> object:
    params = {"x": args.x, "y": args.y, "z": args.z, "resend": args.resend}
    if args.radius is not None:
        params["radius"] = args.radius
    if args.wait:
        frames = await c.drain_until_stream_done(
            lambda: c.request("chunks.set_center", params)
        )
        meshes = sum(1 for f in frames if f.schema_id == 1)
        return {"ok": True, "streamed_frames": len(frames), "meshes": meshes}
    return await c.request("chunks.set_center", params)


async def _cmd_tree(c: EditorClient, args) -> object:
    return await c.request("scene.tree", {})


async def _cmd_stats(c: EditorClient, args) -> object:
    return await c.request("scene.stats", {})


async def _cmd_create(c: EditorClient, args) -> object:
    params: dict = {"kind": args.kind}
    if args.parent is not None:
        params["parent"] = args.parent
    if args.name is not None:
        params["name"] = args.name
    for k in ("x", "y", "z"):
        v = getattr(args, k)
        if v is not None:
            params[k] = v
    return await c.request("scene.create", params)


async def _cmd_rename(c: EditorClient, args) -> object:
    return await c.request("scene.rename", {"id": args.id, "name": args.name})


async def _cmd_reparent(c: EditorClient, args) -> object:
    return await c.request("scene.reparent", {"id": args.id, "parent": args.parent})


async def _cmd_set_transform(c: EditorClient, args) -> object:
    params: dict = {"id": args.id}
    for k in ("px", "py", "pz", "rw", "rx", "ry", "rz", "sx", "sy", "sz"):
        v = getattr(args, k)
        if v is not None:
            params[k] = v
    return await c.request("scene.set_transform", params)


async def _cmd_delete(c: EditorClient, args) -> object:
    return await c.request("scene.delete", {"id": args.id})


async def _cmd_raycast(c: EditorClient, args) -> object:
    params = {"ox": args.ox, "oy": args.oy, "oz": args.oz,
              "dx": args.dx, "dy": args.dy, "dz": args.dz}
    if args.max_distance is not None:
        params["max_distance"] = args.max_distance
    return await c.request("terrain.raycast", params)


async def _cmd_brush(c: EditorClient, args) -> object:
    params = {"shape": args.shape, "mode": args.mode,
              "x": args.x, "y": args.y, "z": args.z}
    for k in ("material", "radius", "hx", "hy", "hz", "height"):
        v = getattr(args, k)
        if v is not None:
            params[k] = v
    # terrain.brush awaits its remesh+broadcast before returning, so the result
    # already implies the meshes were pushed — no stream.done sentinel to wait on
    # (that's a set_center thing). Just return the result.
    return await c.request("terrain.brush", params)


async def _cmd_undo(c: EditorClient, args) -> object:
    return await c.request("edit.undo", {})


async def _cmd_redo(c: EditorClient, args) -> object:
    return await c.request("edit.redo", {})


async def _cmd_rpc(c: EditorClient, args) -> object:
    params = json.loads(args.params) if args.params else {}
    if not isinstance(params, dict):
        raise SystemExit("rpc: --params must be a JSON object")
    return await c.request(args.method, params)


async def _cmd_watch(c: EditorClient, args) -> object:
    """Print every notification + binary-frame header until interrupted."""
    print(f"[watch] listening on port {args.port} (Ctrl+C to stop)", file=sys.stderr)
    c.on_notification = lambda m, p: print(json.dumps({"notification": m, "params": p}))
    c.on_binary = lambda f: print(
        json.dumps({"binary": {"schema_id": f.schema_id, "payload_id": f.payload_id,
                               "bytes": len(f.payload)}})
    )
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.Event().wait()  # run until cancelled (Ctrl+C)
    return {"ok": True}


# Commands that mutate or query an open world need a persistent daemon; running
# them against a freshly-spawned one-shot daemon (no prior world.open) is only
# useful for `open` itself.
_HANDLERS = {
    "open": _cmd_open,
    "save": _cmd_save,
    "ground-lut": _cmd_ground_lut,
    "set-center": _cmd_set_center,
    "tree": _cmd_tree,
    "stats": _cmd_stats,
    "create": _cmd_create,
    "rename": _cmd_rename,
    "reparent": _cmd_reparent,
    "set-transform": _cmd_set_transform,
    "delete": _cmd_delete,
    "raycast": _cmd_raycast,
    "brush": _cmd_brush,
    "undo": _cmd_undo,
    "redo": _cmd_redo,
    "rpc": _cmd_rpc,
    "watch": _cmd_watch,
}


# --------------------------------------------------------------------------- #
# Runner: connect to an existing daemon (--port) or spawn a throwaway one.
# --------------------------------------------------------------------------- #
async def _run_command(args) -> int:
    handler = _HANDLERS[args.command]

    async def go(port: int) -> int:
        c = EditorClient()
        await c.connect(port, args.host)
        try:
            await c.hello("editor-cli")
            result = await handler(c, args)
            _emit(result, args.json)
            return 0
        except RpcRemoteError as e:
            print(f"RPC error {e.code}: {e.rpc_message}", file=sys.stderr)
            return 1
        finally:
            await c.close()

    if args.port:
        return await go(args.port)
    # No --port: spawn a private daemon for this single command.
    async with spawn_daemon(host=args.host) as (_proc, port):
        return await go(port)


# --------------------------------------------------------------------------- #
# `serve`: long-lived daemon + HTTP host for the browser viewport harness.
# --------------------------------------------------------------------------- #
def _start_http(http_port: int, host: str) -> socketserver.TCPServer:
    """Serve editor/extension/ over HTTP (harness HTML + media bundles)."""
    directory = str(_EDITOR_DIR / "extension")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=directory)
    httpd = socketserver.ThreadingTCPServer((host, http_port), handler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


async def _serve(args) -> int:
    daemon = Daemon(host=args.host)
    bound = await daemon.server.start(args.port)
    httpd = _start_http(args.http_port, args.host)
    url = (f"http://{args.host}:{args.http_port}/harness/"
           f"?port={bound}&seed={args.seed}&cam={args.cam}")
    print(f"[serve] daemon ws://{args.host}:{bound}", file=sys.stderr)
    print(f"[serve] harness {url}", file=sys.stderr)
    # The harness URL on stdout so callers (and Chrome MCP) can grab it.
    _emit({"ok": True, "ws_port": bound, "http_port": args.http_port, "harness_url": url},
          args.json)
    try:
        await daemon.server._server.wait_closed()  # type: ignore[union-attr]
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        httpd.shutdown()
        await daemon.server.close()
    return 0


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="editor_client",
                                description="Drive the Fire Editor daemon from the CLI.")
    p.add_argument("--port", type=int, default=0,
                   help=f"daemon port to connect to (0 = spawn a throwaway daemon; "
                        f"use {_DEFAULT_PORT} with `serve`)")
    p.add_argument("--host", default="127.0.0.1", help="daemon host")
    p.add_argument("--json", action="store_true", help="emit raw JSON (machine-readable)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("open", help="open a world by seed or save")
    s.add_argument("--seed", type=int)
    s.add_argument("--save", type=str)

    s = sub.add_parser("save", help="save the open world to a path")
    s.add_argument("--path", required=True)

    sub.add_parser("ground-lut", help="ship the procedural-ground LUT texture frame")

    s = sub.add_parser("set-center", help="stream chunks around a world point")
    s.add_argument("--x", type=float, default=0.0)
    s.add_argument("--y", type=float, default=0.0)
    s.add_argument("--z", type=float, default=24.0)
    s.add_argument("--radius", type=int)
    s.add_argument("--resend", action="store_true", help="restream even already-sent chunks")
    s.add_argument("--wait", action="store_true", help="block until stream.done; report counts")

    sub.add_parser("tree", help="dump the scene-object hierarchy")
    sub.add_parser("stats", help="chunk/mesh counters for the open world")

    s = sub.add_parser("create", help="create a scene object")
    s.add_argument("kind", choices=["empty", "cube", "sphere", "light", "spawn"])
    s.add_argument("--parent", type=int)
    s.add_argument("--name", type=str)
    s.add_argument("--x", type=float)
    s.add_argument("--y", type=float)
    s.add_argument("--z", type=float)

    s = sub.add_parser("rename", help="rename an object")
    s.add_argument("id", type=int)
    s.add_argument("name", type=str)

    s = sub.add_parser("reparent", help="reparent an object (omit --parent to detach to root)")
    s.add_argument("id", type=int)
    s.add_argument("--parent", type=int)

    s = sub.add_parser("set-transform", help="set an object's local TRS (any subset)")
    s.add_argument("id", type=int)
    for k in ("px", "py", "pz", "rw", "rx", "ry", "rz", "sx", "sy", "sz"):
        s.add_argument(f"--{k}", type=float)

    s = sub.add_parser("delete", help="delete an object (and its descendants)")
    s.add_argument("id", type=int)

    s = sub.add_parser("raycast", help="raycast the terrain")
    for k in ("ox", "oy", "oz", "dx", "dy", "dz"):
        s.add_argument(f"--{k}", type=float, required=True)
    s.add_argument("--max-distance", type=float)

    s = sub.add_parser("brush", help="carve/add terrain with a brush")
    s.add_argument("--shape", default="sphere", choices=["sphere", "box", "cylinder"])
    s.add_argument("--mode", default="remove", choices=["add", "remove"])
    s.add_argument("--x", type=float, default=0.0)
    s.add_argument("--y", type=float, default=0.0)
    s.add_argument("--z", type=float, default=0.0)
    s.add_argument("--material", type=int)
    s.add_argument("--radius", type=float)
    s.add_argument("--hx", type=float)
    s.add_argument("--hy", type=float)
    s.add_argument("--hz", type=float)
    s.add_argument("--height", type=float)

    sub.add_parser("undo", help="undo the last edit (terrain or scene)")
    sub.add_parser("redo", help="redo the last undone edit")

    s = sub.add_parser("rpc", help="call an arbitrary RPC method (escape hatch)")
    s.add_argument("method")
    s.add_argument("--params", help="JSON object of params")

    sub.add_parser("watch", help="stream every notification + binary header to stdout")

    s = sub.add_parser("serve", help="run a persistent daemon + HTTP host for the harness")
    s.add_argument("--port", type=int, default=_DEFAULT_PORT, dest="serve_port",
                   help="daemon WebSocket port (0 = OS-assigned)")
    s.add_argument("--http-port", type=int, default=_DEFAULT_HTTP_PORT)
    s.add_argument("--seed", type=int, default=1337, help="seed baked into the harness URL")
    s.add_argument("--cam", default="20,-20,24", help="initial camera, baked into the URL")

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "serve":
        # `serve` has its own port arg (the daemon to *start*, not connect to).
        args.port = args.serve_port
        return asyncio.run(_serve(args))
    try:
        return asyncio.run(_run_command(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
