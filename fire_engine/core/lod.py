"""
core/lod.py — Level-of-Detail policy shared by the World and Terrain layers.

A single ``LODPolicy`` instance is created at boot and shared between:
  - ``world/`` (controls GameObject detail and placement density)
  - ``terrain/`` (controls chunk mesh simplification and streaming radius)

Both layers reading the *same* policy ensures that geometry, placement density,
and shadow detail transition at the same distances — no visual mismatches.

Distance bands
--------------
Band 0 is the nearest (full detail); higher bands are progressively farther
and coarser.  Band assignments are determined by ``band_for(distance_m)``.

Example
-------
    from fire_engine.core.lod import LODPolicy

    policy = LODPolicy(bands=(32.0, 96.0, 192.0, 512.0))
    policy.band_for(20.0)   # → 0  (full detail)
    policy.band_for(50.0)   # → 1
    policy.band_for(200.0)  # → 3
    policy.band_for(999.0)  # → 4  (beyond last band → one past the last index)

Docs: docs/systems/core.md
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LODPolicy:
    """
    Frozen LOD policy defining distance band thresholds in **meters**.

    Attributes
    ----------
    bands : tuple[float, ...]
        Ascending list of distance thresholds in meters.  Objects/chunks closer
        than ``bands[0]`` are in band 0 (full detail); those between ``bands[i-1]``
        and ``bands[i]`` are in band ``i``; those beyond the last threshold are
        in band ``len(bands)``.

    Default bands
    -------------
    (32.0, 96.0, 192.0, 512.0) — roughly:
        Band 0 (<32 m)   : full mesh + all components active
        Band 1 (<96 m)   : slightly reduced placement
        Band 2 (<192 m)  : merged/simplified distant meshes
        Band 3 (<512 m)  : billboard imposters (future)
        Band 4 (≥512 m)  : culled / world-map simulation only

    Example
    -------
    >>> policy = LODPolicy()
    >>> policy.band_for(0.0)
    0
    >>> policy.band_for(95.0)
    1
    >>> policy.band_for(600.0)
    4

    Docs: docs/systems/core.md
    """

    bands: tuple[float, ...] = (32.0, 96.0, 192.0, 512.0)

    def band_for(self, distance_m: float) -> int:
        """
        Return the LOD band index for the given camera distance in **meters**.

        Parameters
        ----------
        distance_m : float — distance from the camera to the object/chunk centre.

        Returns
        -------
        int — 0 (nearest/full-detail) to ``len(bands)`` (beyond last threshold).

        Example
        -------
        >>> LODPolicy(bands=(32.0, 96.0, 192.0)).band_for(100.0)
        2

        Docs: docs/systems/core.md
        """
        for i, threshold in enumerate(self.bands):
            if distance_m < threshold:
                return i
        return len(self.bands)
