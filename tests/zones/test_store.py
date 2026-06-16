"""
tests/zones/test_store.py — unit tests for fire_engine/zones/store.py.

Categories (CLAUDE.md):
- CORRECTNESS: add/remove/get/volumes(), version bumping, tag filtering,
  mark_baseline / get_delta / apply_delta protocol.
- ROUND-TRIP: get_delta -> apply_delta restores the volume set identically.
- DETERMINISM: ZoneStore is non-random; determinism is guaranteed by the
  underlying ZoneVolume primitives.

No panda3d imports (Hard Rule 1).
"""

from __future__ import annotations

from typing import Any

from fire_engine.zones.store import ZoneStore
from fire_engine.zones.volume import ZoneVolume

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store_with_volumes() -> tuple[ZoneStore, ZoneVolume, ZoneVolume]:
    store = ZoneStore()
    a = store.add("grass", (0.0, 0.0, 0.0), (8.0, 8.0, 4.0))
    b = store.add("biome", (0.0, 0.0, 0.0), (50.0, 50.0, 16.0), biome="snow")
    return store, a, b


# ---------------------------------------------------------------------------
# add / remove / get / volumes
# ---------------------------------------------------------------------------


class TestMutationAndQuery:
    def test_add_returns_volume_with_assigned_id(self):
        store = ZoneStore()
        v = store.add("grass", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        assert isinstance(v, ZoneVolume)
        assert v.id == 1

    def test_ids_auto_increment(self):
        store = ZoneStore()
        ids = [
            store.add("grass", (float(i), 0.0, 0.0), (float(i + 1), 1.0, 1.0)).id for i in range(5)
        ]
        assert ids == list(range(1, 6))

    def test_volumes_returns_all_ordered_by_id(self):
        store, a, b = _store_with_volumes()
        assert store.volumes() == (a, b)

    def test_volumes_tag_filter_grass(self):
        store, a, _b = _store_with_volumes()
        assert store.volumes("grass") == (a,)

    def test_volumes_tag_filter_biome(self):
        store, _a, b = _store_with_volumes()
        assert store.volumes("biome") == (b,)

    def test_volumes_tag_no_match_returns_empty(self):
        store, _a, _b = _store_with_volumes()
        assert store.volumes("trees") == ()

    def test_get_existing_returns_volume(self):
        store, a, b = _store_with_volumes()
        assert store.get(a.id) is a
        assert store.get(b.id) is b

    def test_get_missing_returns_none(self):
        store = ZoneStore()
        assert store.get(999) is None

    def test_remove_existing_returns_true(self):
        store, a, _b = _store_with_volumes()
        assert store.remove(a.id) is True

    def test_remove_nonexistent_returns_false(self):
        store = ZoneStore()
        assert store.remove(99) is False

    def test_remove_clears_volume_from_store(self):
        store, a, b = _store_with_volumes()
        store.remove(a.id)
        assert store.volumes() == (b,)
        assert store.get(a.id) is None

    def test_remove_twice_second_is_false(self):
        store, a, _b = _store_with_volumes()
        assert store.remove(a.id) is True
        assert store.remove(a.id) is False


# ---------------------------------------------------------------------------
# version bumping
# ---------------------------------------------------------------------------


class TestVersionBumping:
    def test_initial_version_is_zero(self):
        store = ZoneStore()
        assert store.version == 0

    def test_add_bumps_version(self):
        store = ZoneStore()
        v0 = store.version
        store.add("grass", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        assert store.version > v0

    def test_remove_bumps_version(self):
        store = ZoneStore()
        vol = store.add("grass", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        v1 = store.version
        store.remove(vol.id)
        assert store.version > v1

    def test_remove_nonexistent_does_not_bump_version(self):
        store = ZoneStore()
        v0 = store.version
        store.remove(999)
        assert store.version == v0


# ---------------------------------------------------------------------------
# Saveable protocol: mark_baseline / get_delta / apply_delta
# ---------------------------------------------------------------------------


class TestSaveableProtocol:
    def test_save_key_is_zones(self):
        assert ZoneStore.save_key == "zones"

    def test_baseline_delta_is_empty_dict(self):
        store = ZoneStore()
        store.add("grass", (0.0, 0.0, 0.0), (8.0, 8.0, 4.0))
        store.mark_baseline()
        assert store.get_delta() == {}

    def test_change_after_baseline_produces_nonempty_delta(self):
        store = ZoneStore()
        store.add("grass", (0.0, 0.0, 0.0), (8.0, 8.0, 4.0))
        store.mark_baseline()
        store.add("grass", (20.0, 0.0, 0.0), (28.0, 8.0, 4.0))
        delta = store.get_delta()
        assert delta != {}
        assert "volumes" in delta
        assert "version" in delta

    def test_apply_delta_restores_volumes(self):
        # Build original store + capture delta after a post-baseline change.
        store = ZoneStore()
        store.add("grass", (0.0, 0.0, 0.0), (8.0, 8.0, 4.0), params={"density": 12.0})
        store.mark_baseline()
        store.add("grass", (30.0, 30.0, 6.0), (40.0, 40.0, 10.0))
        delta = store.get_delta()

        # Apply delta onto a fresh store with the same baseline.
        store2 = ZoneStore()
        store2.add("grass", (0.0, 0.0, 0.0), (8.0, 8.0, 4.0), params={"density": 12.0})
        store2.mark_baseline()
        store2.apply_delta(delta)
        assert store2.volumes() == store.volumes()

    def test_apply_empty_delta_is_noop(self):
        store = ZoneStore()
        store.add("grass", (0.0, 0.0, 0.0), (5.0, 5.0, 2.0))
        store.mark_baseline()
        pre_vol = store.volumes()
        pre_ver = store.version
        store.apply_delta({})
        assert store.volumes() == pre_vol
        assert store.version == pre_ver

    def test_apply_future_version_delta_is_ignored(self):
        store = ZoneStore()
        store.add("grass", (0.0, 0.0, 0.0), (5.0, 5.0, 2.0))
        store.mark_baseline()
        pre = store.volumes()
        future: dict[str, Any] = {
            "version": 999,
            "volumes": [ZoneVolume(10, "biome", (0.0, 0.0, 0.0), (100.0, 100.0, 10.0)).to_dict()],
            "next_id": 11,
        }
        store.apply_delta(future)
        assert store.volumes() == pre

    def test_next_id_not_reused_after_round_trip(self):
        store = ZoneStore()
        v1 = store.add("grass", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        store.remove(v1.id)
        store.add("grass", (2.0, 0.0, 0.0), (3.0, 1.0, 1.0))  # id 2
        delta = store.get_delta()

        store2 = ZoneStore()
        store2.apply_delta(delta)
        v3 = store2.add("grass", (4.0, 0.0, 0.0), (5.0, 1.0, 1.0))
        assert v3.id == 3  # ids never reused
