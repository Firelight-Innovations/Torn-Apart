"""pytest config for the Fire Editor daemon tests.

Puts ``editor/`` on ``sys.path`` so ``import fire_editor`` resolves. The repo
root is already added by ``tests/conftest.py`` for ``import torn_apart``.
"""
import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_editor = os.path.join(_root, "editor")
for p in (_root, _editor):
    if p not in sys.path:
        sys.path.insert(0, p)
