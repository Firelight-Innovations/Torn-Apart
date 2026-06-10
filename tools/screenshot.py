"""
tools/screenshot.py — boot the demo, render N frames, save a PNG, exit.

A CI / verification smoke test for the render path: it boots the full demo
(`main.build_demo`), optionally fires a few explosions, sets the game clock /
forces a weather type (sky verification), steps Panda3D's task manager for a
fixed number of frames so terrain streams, lights, and the sky settles,
captures the framebuffer to `tools/out/<name>.png`, and exits cleanly without
entering the blocking main loop.

Usage
-----
    python tools/screenshot.py                       # default demo shot
    python tools/screenshot.py --frames 240 --out spawn.png
    python tools/screenshot.py --explode             # carve a crater first
    # Sky verification shots:
    python tools/screenshot.py --time-of-day 6.5 --weather clear --pitch 5 \
        --out sky/dawn_clear.png
    python tools/screenshot.py --time-of-day 0 --weather clear --pitch 30 \
        --out sky/midnight.png
    python tools/screenshot.py --stub-sky --time-of-day 12 --weather rain \
        --out sky/stub_rain.png    # renderer-only debug (bypasses torn_apart.sky)

Notes
-----
- Requires a graphics pipe (a window is created off to the side; we render
  into it and grab the framebuffer).  On a truly headless box without GL this
  will fail at window creation — that is expected; this is a `window`-class
  tool, not part of the headless pytest suite.
- `--weather` uses `sky_system.weather.force_weather(...)`, which blends over
  20 game-minutes.  The tool anchors the blend ~22 game-minutes before the
  requested `--time-of-day` so the forced weather is FULLY blended in at
  capture time (see `_apply_sky_settings`).
- `--stub-sky` swaps a `types.SimpleNamespace`-based stand-in (all SkyState
  contract fields) into the SkyRendererComponent, so the renderer can be
  debugged in isolation from the headless sky simulation.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import types
from pathlib import Path

# Make the repo root importable when run as `python tools/screenshot.py`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_GAME_DAY_S = 24.0 * 3600.0
# force_weather blends over 20 game-minutes; anchor the override this many
# game-seconds before the capture time so it is fully blended in.
_WEATHER_BLEND_LEAD_S = 1320.0


# ---------------------------------------------------------------------------
# Stub sky (renderer-only debugging; SkyState contract fields)
# ---------------------------------------------------------------------------

# Per-weather presets: coverage, density, fog (1/m), rain, wind m/s, dim.
_STUB_WEATHER = {
    "CLEAR":    (0.18, 0.50, 0.0008, 0.0, 2.0,  1.00),
    "CLOUDY":   (0.45, 0.60, 0.0012, 0.0, 4.0,  0.92),
    "OVERCAST": (0.85, 0.80, 0.0022, 0.0, 5.0,  0.75),
    "FOG":      (0.60, 0.70, 0.0180, 0.0, 1.5,  0.70),
    "RAIN":     (0.80, 0.85, 0.0045, 0.6, 7.0,  0.65),
    "STORM":    (0.95, 1.00, 0.0065, 1.0, 12.0, 0.55),
}


def _make_stub_sky(clock, weather_name: str | None):
    """
    Build a duck-typed stand-in for ``torn_apart.sky.SkySystem``.

    Returns an object with ``update() -> SimpleNamespace``, ``state``, and a
    ``weather`` namespace whose ``force_weather`` is a no-op (the stub's
    weather is fixed by *weather_name*).  The state is recomputed from
    ``clock.game_time_of_day`` on every ``update()`` so ``--time-of-day``
    works, with a crude-but-plausible day cycle.
    """
    from torn_apart.core.math3d import Vec3

    name = (weather_name or "CLEAR").upper()
    cov, den, fog_d, rain, wind, dim = _STUB_WEATHER.get(name, _STUB_WEATHER["CLEAR"])

    def _lerp3(a, b, t):
        return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))

    stub = types.SimpleNamespace()

    def update():
        h = float(clock.game_time_of_day) / 3600.0
        prog = (h - 6.0) / 12.0                       # 0 sunrise .. 1 sunset
        elev = math.sin(max(0.0, min(1.0, prog)) * math.pi)
        daylight = max(0.0, min(1.0, elev * 1.4))
        ang = prog * math.pi
        sun = Vec3(math.cos(ang), 0.25,
                   max(math.sin(ang), -0.4)).normalized()
        moon = Vec3(-sun.x, -0.25, max(-sun.z, 0.15)).normalized()
        warm = max(0.0, 1.0 - elev * 2.2)             # 1 at horizon, 0 high
        sun_color = _lerp3((1.0, 0.96, 0.88), (1.0, 0.55, 0.30), warm)
        zen = _lerp3((0.015, 0.02, 0.045), (0.22, 0.38, 0.62), daylight)
        hor = _lerp3((0.04, 0.05, 0.09), (0.66, 0.74, 0.82), daylight)
        if 0.05 < daylight < 0.55:                    # dawn/dusk warm band
            hor = _lerp3(hor, (0.92, 0.55, 0.34), 0.6)
        gray = (0.5 * daylight + 0.04,) * 3
        stub.state = types.SimpleNamespace(
            sun_dir=sun, moon_dir=moon,
            sun_color=sun_color,
            sun_intensity=daylight * (1.0 - 0.7 * cov) * (0.15 if name == "FOG" else 1.0),
            moon_phase=0.5,
            daylight=daylight,
            star_visibility=max(0.0, min(1.0, (1.0 - daylight * 1.6) * (1.0 - 0.85 * cov))),
            zenith_color=_lerp3(zen, gray, 0.7 * cov),
            horizon_color=_lerp3(hor, gray, 0.7 * cov),
            cloud_coverage=cov, cloud_density=den,
            fog_density=fog_d,
            fog_color=_lerp3(hor, (0.72, 0.74, 0.78), 0.6),
            rain_intensity=rain,
            wind_dir=(0.77, 0.64), wind_speed=wind,
            terrain_light_scale=_lerp3((0.16, 0.19, 0.30), (dim, dim, dim), daylight),
        )
        return stub.state

    stub.update = update
    stub.state = update()
    stub.weather = types.SimpleNamespace(
        current=name, force_weather=lambda w: None)
    stub.clock = clock
    return stub


# ---------------------------------------------------------------------------
# Sky settings (real sky system)
# ---------------------------------------------------------------------------

def _apply_sky_settings(app, time_of_day_h: float | None,
                        weather: str | None) -> None:
    """
    Set the game clock and/or force a weather type on the REAL sky system.

    ``force_weather`` blends over 20 game-minutes anchored at the next
    ``update()``, so: rewind the clock ~22 game-minutes before the target,
    force the weather, step two frames to anchor the blend, then jump to the
    target time (advancing ``game_day`` when the rewind wrapped midnight) —
    the blend window is then fully elapsed at capture time.
    """
    clock = app._clock
    sky = getattr(app, "sky_system", None)

    target_s = (float(time_of_day_h) * 3600.0) % _GAME_DAY_S \
        if time_of_day_h is not None else float(clock.game_time_of_day)

    if weather and sky is not None:
        from torn_apart.sky import WeatherType
        anchor_s = target_s - _WEATHER_BLEND_LEAD_S
        wrapped = anchor_s < 0.0
        clock.game_time_of_day = anchor_s + _GAME_DAY_S if wrapped else anchor_s
        sky.weather.force_weather(WeatherType[weather.upper()])
        app.taskMgr.step()      # anchor the override blend at the rewound time
        app.taskMgr.step()
        if wrapped:
            clock.game_day += 1
    clock.game_time_of_day = target_s


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

def capture(frames: int, out_name: str, explode: bool,
            time_of_day: float | None = None, weather: str | None = None,
            pitch_deg: float = -35.0, yaw_deg: float = 0.0,
            height_m: float | None = None, stub_sky: bool = False) -> Path:
    """
    Build the demo, step `frames` frames, write the framebuffer to a PNG.

    Parameters
    ----------
    frames : int
        Number of frames to step before capturing (gives streaming, lighting,
        and the sky/cloud uniforms time to settle).
    out_name : str
        File name (under tools/out/; may contain subdirs, e.g. "sky/noon.png").
    explode : bool
        If True, fire a downward explosion near spawn before capturing so the
        screenshot shows a crater + the light shaft into it (the "money shot").
    time_of_day : float | None
        Game-clock hour 0–24 (e.g. 6.5 = 06:30).  None keeps the boot time.
    weather : str | None
        Weather name to force (clear/cloudy/overcast/fog/rain/storm); the
        forced weather is fully blended in at capture time.  None = natural.
    pitch_deg : float
        Camera pitch in degrees (negative looks down; default -35 keeps the
        original terrain-centric framing; use ~+5..+30 for sky shots).
    yaw_deg : float
        Camera yaw in degrees around world +Z (0 faces +Y/north; -90 faces
        +X/east — where the sun rises).
    height_m : float | None
        Camera height (world Z, meters).  None keeps the spawn height (10 m);
        use ~110 to fly above the cloud layer (96–104 m).
    stub_sky : bool
        Swap a SimpleNamespace SkyState stub into the SkyRendererComponent
        (renderer-only debugging; bypasses torn_apart.sky entirely).

    Returns
    -------
    Path
        The written PNG path.
    """
    import main as demo
    from torn_apart.core.math3d import Vec3, Quat
    from torn_apart.terrain import SphereBrush, BrushMode, apply_brush

    app = demo.build_demo()

    # Release mouse capture so PHYSICAL mouse movement during the unattended
    # warmup can't feed deltas into the FlyController and swing the camera.
    app.input_state.mouse_captured = False
    app._set_mouse_capture(False)

    # Point the camera so the requested mix of sky/terrain fills the frame.
    app.camera_go.transform.local_rotation = (
        Quat.from_axis_angle(Vec3.UP, math.radians(yaw_deg))
        * Quat.from_axis_angle(Vec3.RIGHT, math.radians(pitch_deg))
    ).normalized()
    if height_m is not None:
        pos = app.camera_go.transform.position
        app.camera_go.transform.position = Vec3(pos.x, pos.y, float(height_m))

    if stub_sky:
        # Renderer-only path: swap the stub into the live SkyRendererComponent.
        from torn_apart.world.sky_renderer import SkyRendererComponent
        if time_of_day is not None:
            app._clock.game_time_of_day = (float(time_of_day) * 3600.0) % _GAME_DAY_S
        stub = _make_stub_sky(app._clock, weather)
        comp = app.sky_go.get_component(SkyRendererComponent)
        comp.sky_system = stub
        app.sky_system = stub
        print("STUB SKY active:", weather or "CLEAR")
    else:
        _apply_sky_settings(app, time_of_day, weather)

    if explode:
        # Carve a crater straight below the camera so the relit interior shows.
        cam = app.camera_go.transform.position
        center = Vec3(cam.x, cam.y + 10.0, cam.z - 8.0)
        apply_brush(
            SphereBrush(3.0),
            center,
            BrushMode.REMOVE,
            material=1,
            chunk_provider=app.chunk_manager.get_or_create,
            bus=app._event_bus,
        )

    # Step the task manager so chunks stream, remesh, relight, and the sky
    # settles.  Each step runs the frame task AND flips the window, so the
    # framebuffer is valid.  Hold the game clock at the requested time so long
    # warmups don't drift the sun (~3 game-minutes per real second otherwise).
    hold_tod = float(app._clock.game_time_of_day)
    for _ in range(frames):
        app.taskMgr.step()
        if time_of_day is not None:
            app._clock.game_time_of_day = hold_tod

    out_dir = _REPO_ROOT / "tools" / "out"
    out_path = out_dir / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Capture the framebuffer into a PNMImage and write it (more reliable than
    # win.save_screenshot, which can no-op if called between flips).
    from panda3d.core import PNMImage, Filename
    img = PNMImage()
    ok = app.win.get_screenshot(img)
    if ok:
        img.write(Filename.from_os_specific(str(out_path)))
    else:
        # Fallback to the ShowBase helper.
        app.screenshot(str(out_path), defaultFilename=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a demo screenshot.")
    parser.add_argument("--frames", type=int, default=180,
                        help="frames to step before capture (default 180)")
    parser.add_argument("--out", default="demo.png",
                        help="output PNG name under tools/out/ (subdirs ok)")
    parser.add_argument("--explode", action="store_true",
                        help="carve a crater before capturing")
    parser.add_argument("--time-of-day", type=float, default=None,
                        metavar="HOURS",
                        help="game-clock hour 0-24 (e.g. 6.5 = 06:30)")
    parser.add_argument("--weather", default=None,
                        choices=["clear", "cloudy", "overcast", "fog",
                                 "rain", "storm"],
                        help="force a weather type (fully blended at capture)")
    parser.add_argument("--pitch", type=float, default=-35.0,
                        help="camera pitch degrees (default -35; use +5..+30 "
                             "for sky shots)")
    parser.add_argument("--yaw", type=float, default=0.0,
                        help="camera yaw degrees (0 = +Y/north, -90 = +X/east)")
    parser.add_argument("--height", type=float, default=None,
                        help="camera height in meters (e.g. 110 = above clouds)")
    parser.add_argument("--stub-sky", action="store_true",
                        help="swap a SkyState stub into the sky renderer "
                             "(renderer-only debugging)")
    args = parser.parse_args()

    path = capture(args.frames, args.out, args.explode,
                   time_of_day=args.time_of_day, weather=args.weather,
                   pitch_deg=args.pitch, yaw_deg=args.yaw,
                   height_m=args.height, stub_sky=args.stub_sky)
    # Report on stdout for CI.
    size = os.path.getsize(path) if path.exists() else 0
    print(f"SCREENSHOT_RESULT wrote {path} ({size} bytes)")
    sys.stdout.flush()
    # Force-exit so the lingering Panda3D window / OpenAL device don't hang us.
    os._exit(0)


if __name__ == "__main__":
    main()
