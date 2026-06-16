"""
tests/test_fly_controller.py — Golden-master / characterisation tests for FlyController.

PURPOSE: Pin CURRENT behaviour. Do NOT fix bugs; report them as comments.
NO panda3d imports; FlyController is declared panda3d-free (player/ layer).

Import path mirrors test_gameobject.py exactly.
InputState is defined in fire_engine.world.app, but that module imports panda3d.
We therefore build a minimal stub (types.SimpleNamespace) rather than importing
InputState — this also avoids the panda3d transitive dependency.

HEADLESS-IMPORT FINDING (see bottom of file): importing FlyController itself
succeeds without panda3d because fire_engine.world.app is gated under
TYPE_CHECKING. However fire_engine.world.component (imported at module level)
DOES import fire_engine.world which may import panda3d depending on __init__.py.
We probe this early with a try/import and skip the whole suite if needed.
"""

from __future__ import annotations

import math
import types

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Headless import check  — skip entire module if panda3d leaks in
# ---------------------------------------------------------------------------
# We do this at collection time so we get one clear skip rather than 35 errors.
try:
    from fire_engine.core.clock import Clock
    from fire_engine.core.event_bus import EventBus
    from fire_engine.core.math3d import Quat, Vec3
    from fire_engine.render.component import Component  # noqa: F401 — probe
    from fire_engine.render.gameobject import GameObject  # noqa: F401
    from fire_engine.render.registry import ComponentRegistry, instantiate
    from fire_engine.simulation.player.fly_controller import _PITCH_LIMIT, FlyController

    _IMPORT_OK = True
    _IMPORT_ERROR = None
except ImportError as _e:
    _IMPORT_OK = False
    _IMPORT_ERROR = str(_e)

pytestmark = pytest.mark.skipif(
    not _IMPORT_OK,
    reason=f"FlyController not headless-importable (panda3d leak?): {_IMPORT_ERROR}",
)

# ---------------------------------------------------------------------------
# Minimal InputState stub — exposes exactly the fields FlyController reads
# ---------------------------------------------------------------------------


def _inp(
    *,
    mouse_captured: bool = False,
    mouse_dx: float = 0.0,
    mouse_dy: float = 0.0,
    move_forward: bool = False,
    move_backward: bool = False,
    move_left: bool = False,
    move_right: bool = False,
    move_up: bool = False,
    move_down: bool = False,
    sprint: bool = False,
) -> types.SimpleNamespace:
    """Build a minimal InputState-compatible stub."""
    return types.SimpleNamespace(
        mouse_captured=mouse_captured,
        mouse_dx=mouse_dx,
        mouse_dy=mouse_dy,
        move_forward=move_forward,
        move_backward=move_backward,
        move_left=move_left,
        move_right=move_right,
        move_up=move_up,
        move_down=move_down,
        sprint=sprint,
    )


# ---------------------------------------------------------------------------
# Fixtures — mirrors test_gameobject.py exactly
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_registry():
    """Isolate each test by clearing the registry before and after."""
    ComponentRegistry.clear()
    yield
    ComponentRegistry.clear()


@pytest.fixture()
def clock():
    return Clock(fixed_dt=0.02, bus=EventBus())


@pytest.fixture()
def go_with_ctrl():
    """A fresh GameObject with a default FlyController already added."""
    go = instantiate()
    ctrl = go.add_component(FlyController)
    return go, ctrl


# ---------------------------------------------------------------------------
# 1. Construction / defaults
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_move_speed(self):
        ctrl = instantiate().add_component(FlyController)
        assert ctrl.move_speed == 10.0

    def test_default_sprint_mult(self):
        ctrl = instantiate().add_component(FlyController)
        assert ctrl.sprint_mult == 5.0

    def test_default_mouse_sensitivity(self):
        ctrl = instantiate().add_component(FlyController)
        assert ctrl.mouse_sensitivity == pytest.approx(0.003)

    def test_params_stored_as_float(self):
        """All three params are coerced through float() — pin this."""
        ctrl = instantiate().add_component(
            FlyController, move_speed=7, sprint_mult=3, mouse_sensitivity=2
        )
        assert isinstance(ctrl.move_speed, float)
        assert isinstance(ctrl.sprint_mult, float)
        assert isinstance(ctrl.mouse_sensitivity, float)

    def test_zero_move_speed_accepted(self):
        """Zero speed — no validation, pin current (no error)."""
        ctrl = instantiate().add_component(FlyController, move_speed=0)
        assert ctrl.move_speed == 0.0

    def test_negative_move_speed_accepted(self):
        """Negative move_speed accepted without validation (pin)."""
        ctrl = instantiate().add_component(FlyController, move_speed=-5.0)
        assert ctrl.move_speed == -5.0

    def test_nan_move_speed_accepted(self):
        """NaN accepted without validation (pin)."""
        ctrl = instantiate().add_component(FlyController, move_speed=float("nan"))
        assert math.isnan(ctrl.move_speed)

    def test_inf_sprint_mult_accepted(self):
        """Inf accepted without validation (pin)."""
        ctrl = instantiate().add_component(FlyController, sprint_mult=float("inf"))
        assert math.isinf(ctrl.sprint_mult)

    def test_initial_yaw_and_pitch_zero(self):
        """yaw and pitch start at 0.0 before awake is called."""
        go = instantiate()
        ctrl = go.add_component(FlyController)
        assert ctrl.yaw == 0.0
        assert ctrl.pitch == 0.0

    def test_initial_input_is_none(self):
        ctrl = instantiate().add_component(FlyController)
        assert ctrl._input is None


# ---------------------------------------------------------------------------
# 2. awake()
# ---------------------------------------------------------------------------


class TestAwake:
    def test_awake_reads_yaw_from_transform_rotation(self, clock):
        """awake() decomposes initial transform rotation into yaw/pitch (pin)."""
        known_yaw = math.radians(45.0)
        q = Quat.from_axis_angle(Vec3.UP, known_yaw)
        go = instantiate(rotation=q)
        ctrl = go.add_component(FlyController)

        clock.update(0.0)
        ComponentRegistry.run_frame(clock)  # triggers awake

        # Pin: yaw is extracted from as_euler()[0] (heading)
        assert ctrl.yaw == pytest.approx(known_yaw, abs=1e-4)

    def test_awake_reads_pitch_from_transform_rotation(self, clock):
        """awake() also captures the pitch component (pin)."""
        known_pitch = math.radians(30.0)
        q = Quat.from_axis_angle(Vec3.RIGHT, known_pitch)
        go = instantiate(rotation=q)
        ctrl = go.add_component(FlyController)

        clock.update(0.0)
        ComponentRegistry.run_frame(clock)

        assert ctrl.pitch == pytest.approx(known_pitch, abs=1e-4)

    def test_awake_no_transform_is_noop(self):
        """If transform is None (bare Component), awake() must not raise (pin)."""
        ctrl = FlyController()
        # Manually call awake() with no game_object/transform attached
        # transform should be None on a bare (not-yet-attached) component.
        # Pin: no exception raised.
        try:
            ctrl.awake()
        except Exception as exc:
            pytest.fail(f"awake() with no transform raised: {exc}")


# ---------------------------------------------------------------------------
# 3. set_input_state()
# ---------------------------------------------------------------------------


class TestSetInputState:
    def test_set_input_state_stores_reference(self):
        ctrl = instantiate().add_component(FlyController)
        inp = _inp()
        ctrl.set_input_state(inp)
        assert ctrl._input is inp

    def test_set_input_state_replaces_previous(self):
        ctrl = instantiate().add_component(FlyController)
        inp1 = _inp(move_forward=True)
        inp2 = _inp(move_backward=True)
        ctrl.set_input_state(inp1)
        ctrl.set_input_state(inp2)
        assert ctrl._input is inp2


# ---------------------------------------------------------------------------
# 4. update() — mouse-look
# ---------------------------------------------------------------------------


class TestMouseLook:
    """Mouse look only changes when mouse_captured=True."""

    def _run_one_frame(self, ctrl, inp, dt=0.016):
        ctrl.set_input_state(inp)
        ctrl.update(dt)

    def test_positive_mouse_dx_decreases_yaw(self, go_with_ctrl):
        """mouse_dx > 0 → yaw decreases (pin sign convention: right = negative yaw delta)."""
        _, ctrl = go_with_ctrl
        ctrl._input = None
        # Manually give a transform (it was just instantiated)
        self._run_one_frame(ctrl, _inp(mouse_captured=True, mouse_dx=10.0))
        # yaw = 0.0 - 10.0 * 0.003 = -0.03
        assert ctrl.yaw == pytest.approx(-0.03, abs=1e-6)

    def test_positive_mouse_dy_decreases_pitch(self, go_with_ctrl):
        """mouse_dy > 0 → pitch decreases (pin sign convention)."""
        _, ctrl = go_with_ctrl
        self._run_one_frame(ctrl, _inp(mouse_captured=True, mouse_dy=10.0))
        assert ctrl.pitch == pytest.approx(-0.03, abs=1e-6)

    def test_mouse_sensitivity_scales_look(self, clock):
        """Sensitivity 0.1 → 10x bigger look delta than default (pin)."""
        go = instantiate()
        ctrl = go.add_component(FlyController, mouse_sensitivity=0.1)
        clock.update(0.016)
        ComponentRegistry.run_frame(clock)  # awake

        ctrl.set_input_state(_inp(mouse_captured=True, mouse_dx=5.0))
        ctrl.update(0.016)
        assert ctrl.yaw == pytest.approx(-0.5, abs=1e-6)

    def test_pitch_clamped_at_positive_limit(self, go_with_ctrl):
        """Large negative mouse_dy cannot drive pitch past +_PITCH_LIMIT."""
        _, ctrl = go_with_ctrl
        self._run_one_frame(ctrl, _inp(mouse_captured=True, mouse_dy=-1_000_000.0))
        assert ctrl.pitch == pytest.approx(_PITCH_LIMIT, abs=1e-6)

    def test_pitch_clamped_at_negative_limit(self, go_with_ctrl):
        """Large positive mouse_dy cannot drive pitch past -_PITCH_LIMIT."""
        _, ctrl = go_with_ctrl
        self._run_one_frame(ctrl, _inp(mouse_captured=True, mouse_dy=1_000_000.0))
        assert ctrl.pitch == pytest.approx(-_PITCH_LIMIT, abs=1e-6)

    def test_yaw_not_clamped_wraps_freely(self, go_with_ctrl):
        """Yaw can exceed 2π without clamping (pin unbounded accumulation)."""
        _, ctrl = go_with_ctrl
        # Spin 100 full revolutions worth of mouse movement
        for _ in range(100):
            self._run_one_frame(ctrl, _inp(mouse_captured=True, mouse_dx=-(2 * math.pi / 0.003)))
        # Yaw should be large, definitely > 2π
        assert abs(ctrl.yaw) > 2 * math.pi

    def test_mouse_uncaptured_no_look_change(self, go_with_ctrl):
        """When mouse_captured=False, yaw/pitch must not change (pin)."""
        _, ctrl = go_with_ctrl
        ctrl.yaw = 1.0
        ctrl.pitch = 0.5
        self._run_one_frame(ctrl, _inp(mouse_captured=False, mouse_dx=50.0, mouse_dy=50.0))
        assert ctrl.yaw == pytest.approx(1.0, abs=1e-9)
        assert ctrl.pitch == pytest.approx(0.5, abs=1e-9)

    def test_zero_sensitivity_no_look_change(self, clock):
        """mouse_sensitivity=0 → yaw/pitch unchanged regardless of deltas (pin)."""
        go = instantiate()
        ctrl = go.add_component(FlyController, mouse_sensitivity=0.0)
        clock.update(0.016)
        ComponentRegistry.run_frame(clock)

        ctrl.set_input_state(_inp(mouse_captured=True, mouse_dx=1000.0, mouse_dy=1000.0))
        ctrl.update(0.016)
        assert ctrl.yaw == pytest.approx(0.0, abs=1e-9)
        assert ctrl.pitch == pytest.approx(0.0, abs=1e-9)

    def test_rotation_has_no_roll(self, go_with_ctrl):
        """After mouse-look the quaternion has no roll component (anti-drift property)."""
        _, ctrl = go_with_ctrl
        # Apply a diagonal mouse delta (both yaw and pitch change)
        ctrl.set_input_state(_inp(mouse_captured=True, mouse_dx=30.0, mouse_dy=-20.0))
        ctrl.update(0.016)

        # Decompose rotation: roll (third component of as_euler) must be ~0
        _h, _p, r = ctrl.game_object.transform.local_rotation.as_euler()
        assert abs(r) < 1e-5, f"Roll drift detected: roll={r}"

    def test_rotation_right_stays_level(self, go_with_ctrl):
        """transform.right must stay in XY plane (Z=0) after yaw-only look."""
        _, ctrl = go_with_ctrl
        ctrl.set_input_state(_inp(mouse_captured=True, mouse_dx=100.0))
        ctrl.update(0.016)

        right = ctrl.game_object.transform.right
        assert abs(right.z) < 1e-5, f"Right vector has Z component after pure yaw: {right}"


# ---------------------------------------------------------------------------
# 5. update() — movement
# ---------------------------------------------------------------------------


class TestMovement:
    """Movement always uses horizontal projections; vertical uses world Z."""

    DT = 0.1  # larger dt makes arithmetic easy to check

    def _go_ctrl(self):
        go = instantiate()
        ctrl = go.add_component(FlyController, move_speed=10.0, sprint_mult=5.0)
        return go, ctrl

    def _run(self, ctrl, inp, dt=None):
        ctrl.set_input_state(inp)
        ctrl.update(dt if dt is not None else self.DT)

    def test_no_keys_no_move(self):
        go, ctrl = self._go_ctrl()
        pos_before = go.transform.local_position
        self._run(ctrl, _inp())
        pos_after = go.transform.local_position
        assert pos_before.approx_eq(pos_after, eps=1e-8)

    def test_forward_moves_in_positive_y(self):
        """Default rotation (yaw=0): forward is +Y (Vec3.FORWARD)."""
        go, ctrl = self._go_ctrl()
        self._run(ctrl, _inp(move_forward=True))
        pos = go.transform.local_position
        # Moved along +Y, X and Z unchanged
        assert pos.y > 0.0
        assert abs(pos.x) < 1e-6
        assert abs(pos.z) < 1e-6

    def test_backward_moves_in_negative_y(self):
        go, ctrl = self._go_ctrl()
        self._run(ctrl, _inp(move_backward=True))
        pos = go.transform.local_position
        assert pos.y < 0.0

    def test_move_right_moves_positive_x(self):
        """Default rotation: right is +X."""
        go, ctrl = self._go_ctrl()
        self._run(ctrl, _inp(move_right=True))
        pos = go.transform.local_position
        assert pos.x > 0.0
        assert abs(pos.y) < 1e-6

    def test_move_left_moves_negative_x(self):
        go, ctrl = self._go_ctrl()
        self._run(ctrl, _inp(move_left=True))
        pos = go.transform.local_position
        assert pos.x < 0.0

    def test_move_up_moves_positive_z(self):
        """Space/E: vertical = world +Z."""
        go, ctrl = self._go_ctrl()
        self._run(ctrl, _inp(move_up=True))
        pos = go.transform.local_position
        assert pos.z > 0.0
        assert abs(pos.x) < 1e-6
        assert abs(pos.y) < 1e-6

    def test_move_down_moves_negative_z(self):
        go, ctrl = self._go_ctrl()
        self._run(ctrl, _inp(move_down=True))
        pos = go.transform.local_position
        assert pos.z < 0.0

    def test_forward_magnitude_correct(self):
        """Pin the exact displacement magnitude: move_speed * dt."""
        go, ctrl = self._go_ctrl()
        self._run(ctrl, _inp(move_forward=True), dt=1.0)
        pos = go.transform.local_position
        # magnitude should equal move_speed * dt = 10 * 1.0 = 10.0
        mag = math.sqrt(pos.x**2 + pos.y**2 + pos.z**2)
        assert mag == pytest.approx(10.0, abs=1e-5)

    def test_sprint_multiplies_speed(self):
        """Sprint flag multiplies displacement by sprint_mult (pin)."""
        go_n, ctrl_n = self._go_ctrl()
        self._run(ctrl_n, _inp(move_forward=True, sprint=False), dt=1.0)

        go_s, ctrl_s = self._go_ctrl()
        self._run(ctrl_s, _inp(move_forward=True, sprint=True), dt=1.0)

        assert go_s.transform.local_position.y == pytest.approx(
            go_n.transform.local_position.y * 5.0, abs=1e-5
        )

    def test_diagonal_move_normalized(self):
        """Forward+Right simultaneously: displacement is normalized then scaled (pin)."""
        go, ctrl = self._go_ctrl()
        self._run(ctrl, _inp(move_forward=True, move_right=True), dt=1.0)
        pos = go.transform.local_position
        mag = math.sqrt(pos.x**2 + pos.y**2 + pos.z**2)
        # Should be 10.0, not 10*sqrt(2)
        assert mag == pytest.approx(10.0, abs=1e-4)

    def test_zero_speed_no_move(self):
        """move_speed=0 → no displacement even with key held (pin)."""
        go = instantiate()
        ctrl = go.add_component(FlyController, move_speed=0.0)
        ctrl.set_input_state(_inp(move_forward=True))
        ctrl.update(0.1)
        pos = go.transform.local_position
        assert pos.approx_eq(Vec3(0, 0, 0), eps=1e-8)

    def test_dt_zero_no_movement(self, go_with_ctrl):
        """dt=0 → movement delta = 0 even with key held; mouse look still applied."""
        go, ctrl = go_with_ctrl
        pos_before = go.transform.local_position
        ctrl.set_input_state(_inp(move_forward=True, mouse_captured=True, mouse_dx=5.0))
        ctrl.update(0.0)
        # Position unchanged
        assert go.transform.local_position.approx_eq(pos_before, eps=1e-8)
        # But yaw DID change (mouse look not gated on dt)
        assert ctrl.yaw != 0.0

    def test_wasd_kills_z_component_when_pitched(self):
        """When pitching steeply, WASD should NOT move along Z (horizontal-only)."""
        go = instantiate()
        ctrl = go.add_component(FlyController, move_speed=10.0)
        # Pitch steeply (looking almost straight up)
        ctrl.pitch = math.radians(80.0)
        ctrl.yaw = 0.0
        ctrl.set_input_state(_inp(move_forward=True))
        ctrl.update(1.0)
        pos = go.transform.local_position
        # Z must not have moved (horizontal projection kills Z)
        assert abs(pos.z) < 1e-4, f"WASD moved along Z when pitched: z={pos.z}"


# ---------------------------------------------------------------------------
# 6. _horizontal() static method
# ---------------------------------------------------------------------------


class TestHorizontal:
    def test_forward_vector_returns_forward(self):
        """Vec3.FORWARD (0,1,0) → normalized on XY = Vec3.FORWARD."""
        result = FlyController._horizontal(Vec3.FORWARD)
        assert result.approx_eq(Vec3.FORWARD, eps=1e-6)

    def test_right_vector_returns_right(self):
        """Vec3.RIGHT (1,0,0) → normalized on XY = Vec3.RIGHT."""
        result = FlyController._horizontal(Vec3.RIGHT)
        assert result.approx_eq(Vec3.RIGHT, eps=1e-6)

    def test_kills_z_component(self):
        """Any non-zero XY vector: Z is zeroed."""
        v = Vec3(1.0, 1.0, 100.0)
        result = FlyController._horizontal(v)
        assert abs(result.z) < 1e-6

    def test_output_is_unit_length(self):
        """Result should always be normalised."""
        v = Vec3(3.0, 4.0, 7.0)
        result = FlyController._horizontal(v)
        mag = math.sqrt(result.x**2 + result.y**2 + result.z**2)
        assert mag == pytest.approx(1.0, abs=1e-6)

    def test_degenerate_straight_up_returns_forward(self):
        """Vec3.UP (0,0,1): XY projection is near-zero → returns Vec3.FORWARD (pin fallback)."""
        result = FlyController._horizontal(Vec3.UP)
        assert result.approx_eq(Vec3.FORWARD, eps=1e-6)

    def test_degenerate_straight_down_returns_forward(self):
        """Vec3 (0,0,-1): same degenerate fallback to Vec3.FORWARD."""
        result = FlyController._horizontal(Vec3(0.0, 0.0, -1.0))
        assert result.approx_eq(Vec3.FORWARD, eps=1e-6)


# ---------------------------------------------------------------------------
# 7. _PITCH_LIMIT constant
# ---------------------------------------------------------------------------


class TestPitchLimit:
    def test_pitch_limit_value(self):
        """_PITCH_LIMIT is exactly math.radians(89) (pin constant)."""
        assert pytest.approx(math.radians(89.0), abs=1e-10) == _PITCH_LIMIT

    def test_pitch_limit_less_than_half_pi(self):
        """_PITCH_LIMIT < π/2 (ensures we never hit the gimbal singularity)."""
        assert math.pi / 2 > _PITCH_LIMIT


# ---------------------------------------------------------------------------
# 8. Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Same inputs over N frames must reproduce identical transform state."""

    def _run_sequence(self, inputs_dts):
        """Run a sequence of (InputState, dt) pairs on a fresh controller; return final state."""
        ComponentRegistry.clear()
        go = instantiate()
        ctrl = go.add_component(
            FlyController, move_speed=10.0, sprint_mult=5.0, mouse_sensitivity=0.003
        )
        for inp, dt in inputs_dts:
            ctrl.set_input_state(inp)
            ctrl.update(dt)
        pos = go.transform.local_position
        rot = go.transform.local_rotation
        ComponentRegistry.clear()
        return (pos.x, pos.y, pos.z), (rot._data[0], rot._data[1], rot._data[2], rot._data[3])

    def test_determinism_same_sequence(self):
        """Two runs with identical input produce identical position and rotation."""
        sequence = [
            (_inp(mouse_captured=True, mouse_dx=5.0, mouse_dy=-2.0, move_forward=True), 0.016),
            (_inp(mouse_captured=True, mouse_dx=-3.0, move_right=True, sprint=True), 0.016),
            (_inp(move_up=True), 0.016),
        ]
        pos1, rot1 = self._run_sequence(sequence)
        pos2, rot2 = self._run_sequence(sequence)

        assert np.allclose(pos1, pos2, atol=1e-8)
        assert np.allclose(rot1, rot2, atol=1e-8)
