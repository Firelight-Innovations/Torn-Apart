"""
tools/verify_weather.py — boot the demo, summon a THUNDERSTORM directly overhead,
fire a lightning strike, and screenshot. Exercises the spatial weather GPU paths
that ``screenshot.py --weather`` cannot (M6 instanced rain gated on the weather-map
precip channel, M7 lightning bolt geometry + shader, M9 cumulonimbus low band).

Run:  python tools/verify_weather.py
Out:  tools/out/weather_verify/thunder.png
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_GAME_DAY_S = 86400.0


def run(
    out_name: str = "weather_verify/thunder.png",
    tod_h: float = 13.0,
    pitch_deg: float = 10.0,
    settle_frames: int = 140,
) -> Path:
    import numpy as np
    import main as demo
    from fire_engine.core.math3d import Vec3, Quat
    from fire_engine.world.weather import CellKind, StormCell
    from fire_engine.core.event_bus import LightningStrikeEvent

    app = demo.build_demo()
    app.input_state.mouse_captured = False
    app._set_mouse_capture(False)

    # Look up so the storm/clouds/bolt fill the frame.
    app.camera_go.transform.local_rotation = (
        Quat.from_axis_angle(Vec3.UP, 0.0)
        * Quat.from_axis_angle(Vec3.RIGHT, math.radians(pitch_deg))
    ).normalized()

    app._clock.game_time_of_day = (float(tod_h) * 3600.0) % _GAME_DAY_S
    hold = float(app._clock.game_time_of_day)
    abs_t = app._clock.game_day * _GAME_DAY_S + hold

    weather = app.sky_system.weather
    cam = app.camera_go.transform.position
    ppos = (float(cam.x), float(cam.y))

    # Inject a full-strength THUNDERSTORM whose center sits ON the player NOW and
    # is at plateau intensity (spawn_time 45% into its lifetime, past the 20% grow
    # ramp, before the 30% decay). center(t)=spawn_pos + (D(t)-D(spawn_time)), so
    # invert for spawn_pos to land the cell overhead despite synoptic drift. This
    # mirrors the M8 devtools "stamp cell at camera" debug key — it gives a fully
    # developed storm immediately rather than waiting ~20 game-min for a summon to
    # ramp + drift in. The cell is a first-class participant (_summoned ∪ natural).
    syn = weather.synoptic
    duration = 6000.0
    spawn_time = abs_t - 0.45 * duration
    drift = np.asarray(syn.displacement(abs_t)) - np.asarray(syn.displacement(spawn_time))
    spawn_pos = (ppos[0] - float(drift[0]), ppos[1] - float(drift[1]))
    cell = StormCell(
        id="v:0",
        kind=CellKind.THUNDERSTORM,
        spawn_time=spawn_time,
        spawn_pos=spawn_pos,
        duration_s=duration,
        radius_m=1100.0,
        peak_intensity=1.0,
        drift_bias=(0.0, 0.0),
    )
    weather._summoned.append(cell)
    chk = cell.center(abs_t, syn)
    print(f"VERIFY injected thunderstorm v:0 center={chk} intensity={cell.intensity(abs_t):.3f}")

    # Settle: stream chunks, build the weather-map precip footprint, fill clouds.
    for _ in range(settle_frames):
        app.taskMgr.step()
        app._clock.game_time_of_day = hold

    # Fire strikes ~150 m IN FRONT of the (north-facing, slightly-up) camera so the
    # descending bolt is framed, building the bolt geometry + compiling
    # lightning.vert/.frag and firing the scene flash. A small fan of seeds raises
    # the odds one bolt is mid-envelope (bright) at capture.
    bus = app._event_bus
    publish = getattr(bus, "publish", None) or bus.publish_deferred
    for i, sx in enumerate((-40.0, 30.0, 110.0)):
        publish(
            LightningStrikeEvent(
                pos=(ppos[0] + sx, ppos[1] + 150.0, cam.z + 210.0),
                ground_pos=(ppos[0] + sx, ppos[1] + 150.0, 8.0),
                seed=20260613 + i,
                time_abs=abs_t,
                cell_id=1,
                intensity=1.0,
            )
        )
    print("VERIFY published 3 LightningStrikeEvents in front of camera")

    # Capture quickly (2 frames) so the bolt is still in its bright leader/return
    # phase rather than faded.
    for _ in range(2):
        app.taskMgr.step()
        app._clock.game_time_of_day = hold

    out_path = _REPO_ROOT / "tools" / "out" / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from panda3d.core import PNMImage, Filename

    img = PNMImage()
    if app.win.get_screenshot(img):
        img.write(Filename.from_os_specific(str(out_path)))
    else:
        app.screenshot(str(out_path), defaultFilename=False)
    print(f"SCREENSHOT_RESULT wrote {out_path}")
    return out_path


if __name__ == "__main__":
    run()
