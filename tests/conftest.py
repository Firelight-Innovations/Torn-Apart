"""
tests/conftest.py — pytest configuration for the Torn Apart test suite.

Ensures the repo root is on sys.path so ``import fire_engine`` works when
pytest is run from the project root directory.
"""

import sys
import os

# Add the project root to sys.path so imports like
# ``from fire_engine.core import Vec3`` resolve correctly.
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
