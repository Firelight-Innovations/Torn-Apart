"""world.screenshot daemon-RPC orchestration tests (no GPU).

The daemon is panda3d-free: ``world.screenshot`` temp-saves the session and spawns
a render subprocess. These tests stub that subprocess (``ChunkService._run_offscreen``)
so they run headless on any box, and assert the orchestration contract:

  * the live session is temp-saved before the render runs,
  * the subprocess argv carries ``--seed <session.seed>`` and the camera pose,
  * a successful render returns ``{ok, path, width, height}`` and cleans the temp
    save dir,
  * a non-zero subprocess exit (or a missing output file) becomes an RpcError.
"""

from __future__ import annotations

import asyncio
import os

import pytest
from fire_editor import Daemon, EditorSession
from fire_editor.rpc import RpcError

_SEED = 1337


def _run(coro):
    return asyncio.run(coro)


def _daemon():
    daemon = Daemon()
    daemon.session = EditorSession.from_seed(_SEED)
    return daemon


def _install_stub(daemon, *, returncode=0, write_png=True, stderr=""):
    """Replace the subprocess seam with a stub; return a dict capturing the call."""
    captured: dict = {}

    async def stub(argv, cwd, env):
        captured["argv"] = list(argv)
        captured["cwd"] = cwd
        captured["env"] = env
        # The save the daemon wrote must exist at render time.
        save_path = argv[argv.index("--save") + 1]
        captured["save_existed"] = os.path.exists(save_path)
        if write_png:
            out_path = argv[argv.index("--out") + 1]
            with open(out_path, "wb") as fh:
                fh.write(b"\x89PNG\r\n\x1a\n stub")
        return returncode, stderr

    daemon.chunks._run_offscreen = stub  # type: ignore[assignment]
    return captured


class TestScreenshotRpc:
    def test_success_returns_path_and_cleans_tempdir(self):
        daemon = _daemon()
        cap = _install_stub(daemon)

        res = _run(
            daemon.chunks.screenshot(
                {"px": 0.0, "py": -20.0, "pz": 12.0, "yaw": 10.0, "pitch": -20.0}
            )
        )

        assert res["ok"] is True
        assert res["width"] == 1280 and res["height"] == 720
        out_path = res["path"]
        assert os.path.exists(out_path), "PNG should be written to the returned path"

        # The session was temp-saved before the render ran...
        assert cap["save_existed"] is True
        save_path = cap["argv"][cap["argv"].index("--save") + 1]
        # ...and its dir is cleaned up afterwards.
        assert not os.path.exists(os.path.dirname(save_path))

        os.remove(out_path)

    def test_argv_carries_session_seed_and_pose(self):
        daemon = _daemon()
        cap = _install_stub(daemon)

        res = _run(
            daemon.chunks.screenshot(
                {"px": 1.0, "py": 2.0, "pz": 3.0, "width": 64, "height": 48, "frames": 3}
            )
        )
        argv = cap["argv"]
        assert argv[:2] == ["-m", "fire_engine.render._impl.offscreen"]
        assert argv[argv.index("--seed") + 1] == str(daemon.session.seed) == str(_SEED)
        assert float(argv[argv.index("--px") + 1]) == 1.0
        assert float(argv[argv.index("--pz") + 1]) == 3.0
        assert argv[argv.index("--width") + 1] == "64"
        assert argv[argv.index("--frames") + 1] == "3"
        # yaw/pitch omitted -> not forwarded (subprocess defaults to look-at-origin).
        assert "--yaw" not in argv and "--pitch" not in argv

        os.remove(res["path"])

    def test_out_path_is_honoured(self, tmp_path):
        daemon = _daemon()
        _install_stub(daemon)
        target = tmp_path / "shot.png"

        res = _run(
            daemon.chunks.screenshot({"px": 0.0, "py": -10.0, "pz": 8.0, "out_path": str(target)})
        )
        assert os.path.abspath(res["path"]) == os.path.abspath(str(target))
        assert target.exists()

    def test_nonzero_exit_raises_rpcerror(self):
        daemon = _daemon()
        _install_stub(daemon, returncode=1, write_png=False, stderr="GL context creation failed")

        with pytest.raises(RpcError) as ei:
            _run(daemon.chunks.screenshot({"px": 0.0, "py": -10.0, "pz": 8.0}))
        assert "GL context creation failed" in str(ei.value)

    def test_missing_output_raises_rpcerror(self):
        daemon = _daemon()
        _install_stub(daemon, returncode=0, write_png=False)

        with pytest.raises(RpcError):
            _run(daemon.chunks.screenshot({"px": 0.0, "py": -10.0, "pz": 8.0}))

    def test_no_world_open_raises(self):
        daemon = Daemon()  # no session
        with pytest.raises(RpcError):
            _run(daemon.chunks.screenshot({"px": 0.0, "py": 0.0, "pz": 8.0}))

    def test_bad_params_raise_invalid_params(self):
        daemon = _daemon()
        _install_stub(daemon)
        with pytest.raises(RpcError):
            _run(daemon.chunks.screenshot({"px": 0.0, "py": 0.0}))  # missing pz
