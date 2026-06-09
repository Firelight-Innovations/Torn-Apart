"""
tests/test_rng.py — Tests for core/rng.py.

Covers:
  - Same keys + same world seed → identical draw sequences
  - Different keys → different streams
  - Different world seeds → different streams
  - Cross-process determinism via subprocess test
    (spawns two separate Python interpreters, asserts identical output)
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from torn_apart.core.rng import set_world_seed, for_domain


# ---------------------------------------------------------------------------
# In-process determinism
# ---------------------------------------------------------------------------

class TestInProcessDeterminism:
    def setup_method(self):
        set_world_seed(1337)

    def test_same_keys_same_stream(self):
        """Two generators with the same keys must produce identical values."""
        g1 = for_domain("terrain", (1, 2, 3))
        g2 = for_domain("terrain", (1, 2, 3))
        a = g1.integers(0, 1_000_000, 10)
        b = g2.integers(0, 1_000_000, 10)
        assert np.array_equal(a, b), "Same keys → identical integers"

    def test_same_keys_same_floats(self):
        g1 = for_domain("procedural", "wasteland_ground")
        g2 = for_domain("procedural", "wasteland_ground")
        a = g1.random(20)
        b = g2.random(20)
        assert np.array_equal(a, b)

    def test_different_keys_different_streams(self):
        g1 = for_domain("terrain", (0, 0, 0))
        g2 = for_domain("terrain", (1, 0, 0))
        a = g1.integers(0, 1_000_000, 10)
        b = g2.integers(0, 1_000_000, 10)
        assert not np.array_equal(a, b), "Different chunk coords → different streams"

    def test_different_domain_names_different_streams(self):
        g1 = for_domain("terrain", (0, 0, 0))
        g2 = for_domain("npc", (0, 0, 0))
        a = g1.integers(0, 1_000_000, 5)
        b = g2.integers(0, 1_000_000, 5)
        assert not np.array_equal(a, b)

    def test_different_seeds_different_streams(self):
        set_world_seed(1337)
        a = for_domain("terrain", (0, 0, 0)).integers(0, 1_000_000, 5)
        set_world_seed(9999)
        b = for_domain("terrain", (0, 0, 0)).integers(0, 1_000_000, 5)
        assert not np.array_equal(a, b), "Different world seeds → different streams"

    def test_various_key_types_accepted(self):
        """Keys may be strings, ints, or nested tuples."""
        set_world_seed(42)
        g1 = for_domain("chunk", 5, (10, 20, 30))
        g2 = for_domain("chunk", 5, (10, 20, 30))
        assert np.array_equal(g1.integers(0, 100, 3), g2.integers(0, 100, 3))


# ---------------------------------------------------------------------------
# Cross-process determinism
# ---------------------------------------------------------------------------

_SUBPROCESS_SCRIPT = """\
import sys
import os
# Ensure the project root is on sys.path so torn_apart imports work.
# The script is executed via -c from cwd=project_root, so '.' works.
sys.path.insert(0, os.getcwd())
from torn_apart.core.rng import set_world_seed, for_domain
set_world_seed({seed})
rng = for_domain("terrain", (1, 2, 3))
result = rng.integers(0, 1_000_000, 5)
print(list(result))
"""


def _run_rng_subprocess(seed: int) -> list[int]:
    """
    Spawn a fresh Python interpreter, run the RNG script, and return the
    list of generated integers.

    This proves that the blake2b key digest (not Python's salted hash())
    is used, since hash() would produce different values on each process run.
    """
    # Locate the project root (two directories up from this test file)
    project_root = str(
        __import__("pathlib").Path(__file__).parent.parent.resolve()
    )
    script = _SUBPROCESS_SCRIPT.format(seed=seed)

    # Write the script inline; use sys.executable so we use the same venv
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=project_root,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Subprocess failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
    return eval(result.stdout.strip())


class TestCrossProcessDeterminism:
    """
    Spawns two separate Python interpreter processes and verifies that
    for_domain produces the SAME output — proving blake2b stability.

    If Python's built-in hash() were used, the per-process salt would make
    these differ, causing silent world-gen divergence across save/load cycles.
    """

    def test_two_processes_identical_output(self):
        seed = 1337
        run1 = _run_rng_subprocess(seed)
        run2 = _run_rng_subprocess(seed)
        assert run1 == run2, (
            f"Cross-process RNG mismatch!\n"
            f"Run 1: {run1}\n"
            f"Run 2: {run2}\n"
            "This means the key digest is NOT stable (hash() salting detected)."
        )

    def test_different_seeds_different_processes_differ(self):
        run_a = _run_rng_subprocess(1337)
        run_b = _run_rng_subprocess(9999)
        assert run_a != run_b, "Different seeds must produce different streams"

    def test_output_length_correct(self):
        """Subprocess produces exactly 5 integers."""
        result = _run_rng_subprocess(42)
        assert len(result) == 5
        assert all(0 <= v < 1_000_000 for v in result)
