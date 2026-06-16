"""
tests/lighting/_impl/test_protocols.py — Headless tests for
fire_engine.lighting._impl.protocols.

Covers:
- GeometryOccupancyProvider is exported and is a runtime-checkable Protocol.
- A conforming class is accepted by isinstance.
- A non-conforming class is rejected.
- Re-export from fire_engine.lighting.volume gives the same object.

No panda3d imports.
"""

from __future__ import annotations

import numpy as np

from fire_engine.lighting._impl.protocols import GeometryOccupancyProvider


class _ConformingProvider:
    """Minimal correct implementation of GeometryOccupancyProvider."""

    def rasterize_occupancy(
        self,
        origin_cell: tuple[int, int, int],
        cells: int,
        cell_m: float,
        albedo_occ: np.ndarray,
        emission: np.ndarray,
    ) -> None:
        pass  # no-op for tests


class _MissingMethodProvider:
    """Class that does NOT implement rasterize_occupancy."""

    pass


class TestProtocolStructure:
    def test_protocol_is_importable(self):
        from fire_engine.lighting._impl.protocols import GeometryOccupancyProvider as GOP

        assert GOP is GeometryOccupancyProvider

    def test_protocol_is_runtime_checkable(self):
        """isinstance checks must work without raising TypeError."""
        obj = _ConformingProvider()
        result = isinstance(obj, GeometryOccupancyProvider)
        assert isinstance(result, bool)

    def test_conforming_class_passes_isinstance(self):
        assert isinstance(_ConformingProvider(), GeometryOccupancyProvider)

    def test_non_conforming_class_fails_isinstance(self):
        assert not isinstance(_MissingMethodProvider(), GeometryOccupancyProvider)

    def test_plain_object_fails_isinstance(self):
        assert not isinstance(object(), GeometryOccupancyProvider)


class TestReExport:
    def test_reexported_from_volume(self):
        from fire_engine.lighting.volume import GeometryOccupancyProvider as GopFromVolume

        assert GopFromVolume is GeometryOccupancyProvider


class TestConformingProvider:
    def test_rasterize_occupancy_callable(self):
        prov = _ConformingProvider()
        ao = np.zeros((8, 8, 8, 4), dtype=np.uint8)
        em = np.zeros((8, 8, 8, 4), dtype=np.uint8)
        # Must be callable without error.
        prov.rasterize_occupancy((0, 0, 0), 8, 0.5, ao, em)

    def test_rasterize_occupancy_can_write_cells(self):
        """A provider that writes a cell is structurally valid."""

        class _WritingProvider:
            def rasterize_occupancy(self, origin_cell, cells, cell_m, albedo_occ, emission):
                albedo_occ[1, 2, 3, 3] = 200

        prov = _WritingProvider()
        assert isinstance(prov, GeometryOccupancyProvider)
        ao = np.zeros((8, 8, 8, 4), dtype=np.uint8)
        em = np.zeros((8, 8, 8, 4), dtype=np.uint8)
        prov.rasterize_occupancy((0, 0, 0), 8, 0.5, ao, em)
        assert ao[1, 2, 3, 3] == 200
