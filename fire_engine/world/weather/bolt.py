"""
weather/bolt.py — Procedural stepped-leader lightning bolt geometry (M7, headless).

A lightning bolt is grown as a **stepped leader**: starting at the cloud base it
fans ``K`` candidate directions in a downward-biased cone each step, scores each
by downward progress minus a seeded value-noise "air resistance" minus repulsion
from the channel grown so far, and picks one with a softmax (so the path wanders
believably without ever folding back).  The first channel to reach the ground
becomes the bright **return stroke**; side branches spawn with a small per-step
probability, dim with depth, and terminate in mid-air (real branches rarely reach
the ground).

The whole thing is a pure function of ``seed``: ``generate_bolt(seed, ...)`` draws
every random number from ``for_domain("weather", "bolt", seed, ...)``, so the same
strike renders byte-identically on every machine and after a save/load (the
:class:`~fire_engine.core.event_bus.LightningStrikeEvent` carries only the seed;
the renderer regrows the geometry).

The output is a flat list of line **segments** packed into numpy arrays — the
render half (``world/lightning_renderer.py``) expands each segment into a
camera-facing ribbon in its vertex shader.  This module never touches panda3d
(Hard Rule 1).

Performance
-----------
The single stepped-leader growth loop is the ONE bounded Python loop the M7 spec
allows (``config.bolt_max_steps`` ≤ 400 steps total across the main channel + all
branches).  Each step is a handful of numpy ops over ``K`` candidates and the
existing channel — target < 5 ms for a full bolt (see ``tests/``).

Units: meters, world Z-up.

Example
-------
    from fire_engine.core import load_config, set_world_seed
    from fire_engine.world.weather.bolt import generate_bolt

    set_world_seed(1337)
    bolt = generate_bolt(seed=42, start=(0.0, 0.0, 220.0), ground_z=8.0,
                         config=load_config())
    bolt.a.shape        # (N, 3) segment start points
    bolt.is_main.any()  # True — at least one channel reached the ground

Docs: docs/systems/world.weather.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fire_engine.core.config import Config
from fire_engine.core.rng import for_domain

__all__ = ["BoltGeometry", "generate_bolt"]


@dataclass(frozen=True)
class BoltGeometry:
    """
    Packed line-segment geometry for one lightning bolt.

    Every channel (the main return stroke + every side branch) is decomposed
    into straight segments; the arrays below hold one row per segment, so the
    renderer uploads ``a``/``b`` as a line list and expands each to a
    camera-facing ribbon whose width/brightness come from ``width``/
    ``brightness``.

    Attributes
    ----------
    a : np.ndarray — ``(N, 3)`` float32 segment start points, world meters.
    b : np.ndarray — ``(N, 3)`` float32 segment end points, world meters.
    width : np.ndarray — ``(N,)`` float32 ribbon half-width per segment (m);
        the main channel is wider, branches thinner with depth.
    brightness : np.ndarray — ``(N,)`` float32 0–1+ emissive scale per segment
        (the main return stroke is brightest; branches dim with depth).
    is_main : np.ndarray — ``(N,)`` bool: True for the main return-stroke
        channel (the one that reached ``ground_z``), False for side branches.

    Invariants
    ----------
    * All five arrays share the same length ``N`` (``len(bolt)``).
    * The main channel's last segment endpoint ``b`` is at (or within a step of)
      ``ground_z``; branch endpoints stay above ground.

    Docs: docs/systems/world.weather.md
    """

    a: np.ndarray
    b: np.ndarray
    width: np.ndarray
    brightness: np.ndarray
    is_main: np.ndarray

    def __len__(self) -> int:
        return int(self.a.shape[0])


def _lattice_hash_vec(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, salt: int) -> np.ndarray:
    """
    Deterministic integer hash of lattice corners → [0, 1), vectorised.

    A pure-arithmetic spl-mix style hash (no per-call RNG construction), cheap
    enough to call eight times per growth step over all K candidates.  The
    ``salt`` (drawn ONCE per bolt from ``for_domain`` — see :func:`generate_bolt`)
    folds in the world seed + bolt seed, keeping the field cross-process stable
    (Hard Rule 2: the salt is the sole entropy source for the field).
    """
    # uint64 arithmetic wraps modulo 2**64 (silently, like a C hash mix).
    h = (
        np.uint64(salt)
        ^ (ix.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15))
        ^ (iy.astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F))
        ^ (iz.astype(np.uint64) * np.uint64(0x165667B19E3779F9))
    )
    h ^= h >> np.uint64(30)
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(27)
    h *= np.uint64(0x94D049BB133111EB)
    h ^= h >> np.uint64(31)
    result: np.ndarray = h.astype(np.float64) / float(1 << 64)
    return result


def _value_noise_3d_vec(pts: np.ndarray, salt: int) -> np.ndarray:
    """
    Vectorised value-noise in [0, 1] at every point in ``pts`` (shape ``(K, 3)``).

    Trilinear interpolation of per-lattice-corner integer hashes (the "air
    resistance" field).  Pure function of (``salt``, quantised positions); fully
    numpy (no Python per-point loop) so it is cheap inside the growth loop.
    """
    cell = 6.0
    q = pts / cell
    f = np.floor(q)
    t = q - f
    w = t * t * (3.0 - 2.0 * t)
    fi = f.astype(np.int64)
    fx, fy, fz = fi[:, 0], fi[:, 1], fi[:, 2]
    one = np.int64(1)

    c000 = _lattice_hash_vec(fx, fy, fz, salt)
    c100 = _lattice_hash_vec(fx + one, fy, fz, salt)
    c010 = _lattice_hash_vec(fx, fy + one, fz, salt)
    c110 = _lattice_hash_vec(fx + one, fy + one, fz, salt)
    c001 = _lattice_hash_vec(fx, fy, fz + one, salt)
    c101 = _lattice_hash_vec(fx + one, fy, fz + one, salt)
    c011 = _lattice_hash_vec(fx, fy + one, fz + one, salt)
    c111 = _lattice_hash_vec(fx + one, fy + one, fz + one, salt)

    wx, wy, wz = w[:, 0], w[:, 1], w[:, 2]
    x00 = c000 + (c100 - c000) * wx
    x10 = c010 + (c110 - c010) * wx
    x01 = c001 + (c101 - c001) * wx
    x11 = c011 + (c111 - c011) * wx
    y0 = x00 + (x10 - x00) * wy
    y1 = x01 + (x11 - x01) * wy
    result: np.ndarray = y0 + (y1 - y0) * wz
    return result


def _fan_directions(
    forward: np.ndarray, k: int, cone_rad: float, rng: np.random.Generator
) -> np.ndarray:
    """
    ``k`` candidate unit directions fanned around ``forward`` within a cone.

    The first candidate is ``forward`` itself (so "keep going straight" is always
    on the table); the rest are uniformly sampled inside the cone (uniform in
    ``cosθ`` for an even solid-angle spread).  Vectorised — no per-candidate loop.

    Parameters
    ----------
    forward : np.ndarray — shape ``(3,)`` unit forward direction.
    k : int — number of candidates (≥ 1).
    cone_rad : float — cone half-angle in radians.
    rng : np.random.Generator — the seeded stream for this step.

    Returns
    -------
    np.ndarray — shape ``(k, 3)`` unit candidate directions.
    """
    # Build an orthonormal basis (forward, u, v).
    fwd = forward / (np.linalg.norm(forward) + 1e-9)
    ref = np.array([0.0, 0.0, 1.0]) if abs(fwd[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(ref, fwd)
    u /= np.linalg.norm(u) + 1e-9
    v = np.cross(fwd, u)

    cos_min = math.cos(cone_rad)
    # First candidate = straight ahead; remaining sampled in the cone.
    n_rand = max(k - 1, 0)
    cos_t = rng.uniform(cos_min, 1.0, size=n_rand)
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t * cos_t))
    phi = rng.uniform(0.0, 2.0 * math.pi, size=n_rand)
    dirs_rand = (
        cos_t[:, None] * fwd[None, :]
        + (sin_t * np.cos(phi))[:, None] * u[None, :]
        + (sin_t * np.sin(phi))[:, None] * v[None, :]
    )
    out = np.empty((k, 3), dtype=np.float64)
    out[0] = fwd
    if n_rand:
        out[1:] = dirs_rand
    # Normalise defensively.
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
    return out


def _grow_channel(
    start: np.ndarray,
    forward: np.ndarray,
    ground_z: float,
    seed: int,
    noise_salt: int,
    branch_id: int,
    depth: int,
    cfg: Config,
    channel_pts: list[np.ndarray],
    budget: list[int],
    out_a: list[np.ndarray],
    out_b: list[np.ndarray],
    out_w: list[float],
    out_br: list[float],
    out_main: list[bool],
    reached: list[bool],
) -> None:
    """
    Grow one stepped-leader channel from ``start``, recording its segments.

    The recursion depth is the *branch* depth (small, ≤ ``bolt_branch_max_depth``
    + the main channel), NOT the per-step count — every channel shares one global
    ``budget`` of ``bolt_max_steps`` steps so the total stays bounded (the one
    allowed loop).  The main channel (``depth == 0``) is the only one that may
    reach the ground; branches stop in mid-air at a fraction of the remaining
    drop.

    This appends to the ``out_*`` lists in place (segment soup, flattened later).
    """
    k = int(cfg.bolt_candidates)
    cone = math.radians(float(cfg.bolt_cone_deg))
    temp = float(cfg.bolt_softmax_temp)
    step_lo = float(cfg.bolt_step_len_min_m)
    step_hi = float(cfg.bolt_step_len_max_m)
    branch_p = float(cfg.bolt_branch_prob)
    noise_gain = float(cfg.bolt_noise_gain)
    repel_gain = float(cfg.bolt_repulsion_gain)
    max_depth = int(cfg.bolt_branch_max_depth)

    is_main = depth == 0
    pos = start.astype(np.float64).copy()
    fwd = forward.astype(np.float64).copy()

    # Width / brightness decay with branch depth (the main stroke is the boldest).
    base_w = 0.9 if is_main else 0.45 * (0.6**depth)
    base_br = 1.6 if is_main else 0.55 * (0.6**depth)

    # Branches terminate after covering a random fraction of the remaining drop.
    branch_rng = for_domain("weather", "bolt", int(seed), "branch", int(branch_id))
    if is_main:
        stop_z = ground_z
    else:
        drop = max(0.0, float(pos[2]) - ground_z)
        stop_z = float(pos[2]) - drop * float(branch_rng.uniform(0.12, 0.5))

    step_i = 0
    while budget[0] > 0:
        if pos[2] <= stop_z:
            if is_main:
                reached[0] = True
            break
        budget[0] -= 1
        rng = for_domain("weather", "bolt", int(seed), "step", int(branch_id), int(step_i))

        # Bias forward toward "down" so the leader trends to ground.
        biased = fwd + np.array([0.0, 0.0, -1.0]) * 0.6
        biased /= np.linalg.norm(biased) + 1e-9
        cand = _fan_directions(biased, k, cone, rng)  # (k, 3)
        step_len = float(rng.uniform(step_lo, step_hi))
        next_pts = pos[None, :] + cand * step_len  # (k, 3)

        # Score each candidate: downward progress − air resistance − repulsion.
        down = -(next_pts[:, 2] - pos[2])  # +ve = descends
        # Seeded value-noise "air resistance" at each candidate endpoint (one
        # vectorised pass over all K candidates — no per-candidate Python loop).
        resist = _value_noise_3d_vec(next_pts, noise_salt)
        # Repulsion from the channel grown so far (cheap inverse-distance to the
        # nearest few recorded points) — keeps branches from overlapping.
        if channel_pts:
            recent = np.asarray(channel_pts[-24:])  # (M, 3)
            d2 = ((next_pts[:, None, :] - recent[None, :, :]) ** 2).sum(axis=2)
            repel = 1.0 / (d2.min(axis=1) + 1.0)  # (k,)
        else:
            repel = np.zeros(k)

        score = down - noise_gain * resist * step_len - repel_gain * repel * step_len
        # Softmax pick (temperature `temp`); seeded categorical draw.
        z = (score - score.max()) / max(temp, 1e-3)
        p = np.exp(z)
        p /= p.sum()
        choice = int(rng.choice(k, p=p))
        new_pos = next_pts[choice]
        chosen_dir = cand[choice]

        # Record the segment.
        out_a.append(pos.copy())
        out_b.append(new_pos.copy())
        out_w.append(base_w)
        out_br.append(base_br)
        out_main.append(is_main)
        channel_pts.append(new_pos.copy())

        # Maybe spawn a side branch from the new vertex (not from branches past
        # the max depth — keeps the recursion + step budget bounded).
        if depth < max_depth and budget[0] > 0 and float(rng.random()) < branch_p:
            # Branch heads off at an angle from the parent forward direction.
            boff = _fan_directions(chosen_dir, 2, math.radians(55.0), rng)[1]
            _grow_channel(
                new_pos,
                boff,
                ground_z,
                seed,
                noise_salt,
                branch_id * 97 + step_i + 1,
                depth + 1,
                cfg,
                channel_pts,
                budget,
                out_a,
                out_b,
                out_w,
                out_br,
                out_main,
                reached,
            )

        pos = new_pos
        fwd = chosen_dir
        step_i += 1


def generate_bolt(
    seed: int,
    start: tuple[float, float, float],
    ground_z: float,
    config: Config,
) -> BoltGeometry:
    """
    Grow a deterministic stepped-leader lightning bolt from ``start`` to ground.

    Pure function of (world seed, ``seed``): the same arguments always produce
    byte-identical geometry (the strike event carries only ``seed``; the renderer
    regrows the bolt).  All randomness flows through
    ``for_domain("weather", "bolt", seed, ...)`` (Hard Rule 2).

    Parameters
    ----------
    seed : int — the bolt RNG seed (from the strike event).
    start : tuple[float, float, float] — cloud-base origin world XYZ (meters);
        the top of the bolt.
    ground_z : float — world Z (meters) the main channel grows down to (the
        ground or roof height under the strike XY).
    config : Config — reads the ``bolt_*`` tuning fields.

    Returns
    -------
    BoltGeometry — packed segment arrays (see the class).  The main channel
    reaches ``ground_z`` within one step length unless the step budget is
    exhausted first (extremely tall clouds + tiny steps); branches stop above
    ground.

    Example
    -------
    >>> from fire_engine.core import load_config, set_world_seed
    >>> set_world_seed(1337)
    >>> b = generate_bolt(7, (0.0, 0.0, 200.0), 8.0, load_config())
    >>> len(b) > 0 and b.is_main.any()
    True

    Docs: docs/systems/world.weather.md
    """
    start_arr = np.asarray(start, dtype=np.float64)
    gz = float(ground_z)

    out_a: list[np.ndarray] = []
    out_b: list[np.ndarray] = []
    out_w: list[float] = []
    out_br: list[float] = []
    out_main: list[bool] = []
    channel_pts: list[np.ndarray] = []
    budget = [int(config.bolt_max_steps)]
    reached = [False]

    # Draw the value-noise hash salt ONCE from the seeded RNG (folds in the world
    # seed + bolt seed cross-process-stably); the per-corner lattice hash is then
    # pure arithmetic — fast enough for the growth loop (Hard Rule 2 satisfied:
    # the salt is the sole entropy source for the field).
    noise_salt = int(for_domain("weather", "bolt", int(seed), "noise_salt").integers(0, 1 << 63))

    _grow_channel(
        start_arr,
        np.array([0.0, 0.0, -1.0]),
        gz,
        int(seed),
        noise_salt,
        branch_id=0,
        depth=0,
        cfg=config,
        channel_pts=channel_pts,
        budget=budget,
        out_a=out_a,
        out_b=out_b,
        out_w=out_w,
        out_br=out_br,
        out_main=out_main,
        reached=reached,
    )

    if not out_a:  # degenerate (start already at/below ground)
        z = np.zeros((0, 3), dtype=np.float32)
        e = np.zeros((0,), dtype=np.float32)
        return BoltGeometry(z, z.copy(), e, e.copy(), e.astype(bool))

    a = np.asarray(out_a, dtype=np.float32)
    b = np.asarray(out_b, dtype=np.float32)
    width = np.asarray(out_w, dtype=np.float32)
    brightness = np.asarray(out_br, dtype=np.float32)
    is_main = np.asarray(out_main, dtype=bool)

    # If the main channel reached ground, snap its final endpoint exactly to
    # ground_z so the strike point is well-defined for the flash light.
    if reached[0]:
        main_idx = np.where(is_main)[0]
        if main_idx.size:
            b[main_idx[-1], 2] = np.float32(gz)

    return BoltGeometry(a=a, b=b, width=width, brightness=brightness, is_main=is_main)
