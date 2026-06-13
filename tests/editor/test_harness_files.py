"""Harness wiring guards (EDITOR_PRD agent access).

Cheap structural checks that the browser viewport harness references the bundles
it actually needs, and that the CLI exposes the agent-facing subcommands. These
don't boot a browser — they just keep the static wiring from rotting.
"""
from __future__ import annotations

import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_EXT = os.path.join(_ROOT, "editor", "extension")


def _read(*parts: str) -> str:
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return f.read()


class TestHarnessHtml:
    def test_html_references_both_bundles(self):
        html = _read(_EXT, "harness", "index.html")
        # harnessBoot must load BEFORE sceneView (it installs the host shim and
        # injects the markup that sceneView.ts dereferences at module load).
        boot = html.index("harnessBoot.js")
        view = html.index("sceneView.js")
        assert boot < view, "harnessBoot.js must be referenced before sceneView.js"

    def test_bundles_built_or_skip(self):
        media = os.path.join(_EXT, "media")
        needed = ["harnessBoot.js", "sceneView.js"]
        missing = [b for b in needed if not os.path.exists(os.path.join(media, b))]
        if missing:
            pytest.skip(f"extension not built (missing {missing}); run "
                        f"`npm run compile` in editor/extension")
        for b in needed:
            assert os.path.getsize(os.path.join(media, b)) > 0


class TestCli:
    def test_cli_imports_and_lists_commands(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "editor_client_cli", os.path.join(_ROOT, "tools", "editor_client.py")
        )
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Every RPC-mapped subcommand the agent harness doc advertises.
        for cmd in ("open", "save", "set-center", "tree", "create", "rename",
                    "reparent", "set-transform", "delete", "raycast", "brush",
                    "undo", "redo", "rpc", "watch"):
            assert cmd in mod._HANDLERS, f"CLI missing subcommand: {cmd}"

    def test_cli_parser_builds(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "editor_client_cli2", os.path.join(_ROOT, "tools", "editor_client.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        parser = mod._build_parser()
        # `serve` is the harness host command and must parse with its own ports.
        ns = parser.parse_args(["serve", "--port", "8123", "--http-port", "8770"])
        assert ns.command == "serve" and ns.serve_port == 8123 and ns.http_port == 8770
