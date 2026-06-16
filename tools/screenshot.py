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
        --out sky/stub_rain.png    # renderer-only debug (bypasses fire_engine.world.sky)

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
    "CLEAR": (0.18, 0.50, 0.0008, 0.0, 2.0, 1.00),
    "CLOUDY": (0.45, 0.60, 0.0012, 0.0, 4.0, 0.92),
    "OVERCAST": (0.85, 0.80, 0.0022, 0.0, 5.0, 0.75),
    "FOG": (0.60, 0.70, 0.0180, 0.0, 1.5, 0.70),
    "RAIN": (0.80, 0.85, 0.0045, 0.6, 7.0, 0.65),
    "STORM": (0.95, 1.00, 0.0065, 1.0, 12.0, 0.55),
}


def _make_stub_sky(clock, weather_name: str | None):
    """
    Build a duck-typed stand-in for ``fire_engine.world.sky.SkySystem``.

    Returns an object with ``update() -> SimpleNamespace``, ``state``, and a
    ``weather`` namespace whose ``force_weather`` is a no-op (the stub's
    weather is fixed by *weather_name*).  The state is recomputed from
    ``clock.game_time_of_day`` on every ``update()`` so ``--time-of-day``
    works, with a crude-but-plausible day cycle.
    """
    from fire_engine.core.math3d import Vec3

    name = (weather_name or "CLEAR").upper()
    cov, den, fog_d, rain, wind, dim = _STUB_WEATHER.get(name, _STUB_WEATHER["CLEAR"])

    def _lerp3(a, b, t):
        return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))

    stub = types.SimpleNamespace()

    def update():
        h = float(clock.game_time_of_day) / 3600.0
        prog = (h - 6.0) / 12.0  # 0 sunrise .. 1 sunset
        elev = math.sin(max(0.0, min(1.0, prog)) * math.pi)
        daylight = max(0.0, min(1.0, elev * 1.4))
        ang = prog * math.pi
        sun = Vec3(math.cos(ang), 0.25, max(math.sin(ang), -0.4)).normalized()
        moon = Vec3(-sun.x, -0.25, max(-sun.z, 0.15)).normalized()
        warm = max(0.0, 1.0 - elev * 2.2)  # 1 at horizon, 0 high
        sun_color = _lerp3((1.0, 0.96, 0.88), (1.0, 0.55, 0.30), warm)
        zen = _lerp3((0.015, 0.02, 0.045), (0.22, 0.38, 0.62), daylight)
        hor = _lerp3((0.04, 0.05, 0.09), (0.66, 0.74, 0.82), daylight)
        if 0.05 < daylight < 0.55:  # dawn/dusk warm band
            hor = _lerp3(hor, (0.92, 0.55, 0.34), 0.6)
        gray = (0.5 * daylight + 0.04,) * 3
        stub.state = types.SimpleNamespace(
            sun_dir=sun,
            moon_dir=moon,
            sun_color=sun_color,
            sun_intensity=daylight * (1.0 - 0.7 * cov) * (0.15 if name == "FOG" else 1.0),
            moon_phase=0.5,
            daylight=daylight,
            star_visibility=max(0.0, min(1.0, (1.0 - daylight * 1.6) * (1.0 - 0.85 * cov))),
            zenith_color=_lerp3(zen, gray, 0.7 * cov),
            horizon_color=_lerp3(hor, gray, 0.7 * cov),
            cloud_coverage=cov,
            cloud_density=den,
            fog_density=fog_d,
            fog_color=_lerp3(hor, (0.72, 0.74, 0.78), 0.6),
            rain_intensity=rain,
            wind_dir=(0.77, 0.64),
            wind_speed=wind,
            terrain_light_scale=_lerp3((0.16, 0.19, 0.30), (dim, dim, dim), daylight),
        )
        return stub.state

    stub.update = update
    stub.state = update()
    stub.weather = types.SimpleNamespace(current=name, force_weather=lambda w: None)
    stub.clock = clock
    return stub


# ---------------------------------------------------------------------------
# Sky settings (real sky system)
# ---------------------------------------------------------------------------


def _apply_sky_settings(app, time_of_day_h: float | None, weather: str | None) -> None:
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

    target_s = (
        (float(time_of_day_h) * 3600.0) % _GAME_DAY_S
        if time_of_day_h is not None
        else float(clock.game_time_of_day)
    )

    if weather and sky is not None:
        from fire_engine.world.sky import WeatherType

        anchor_s = target_s - _WEATHER_BLEND_LEAD_S
        wrapped = anchor_s < 0.0
        clock.game_time_of_day = anchor_s + _GAME_DAY_S if wrapped else anchor_s
        sky.weather.force_weather(WeatherType[weather.upper()])
        app.taskMgr.step()  # anchor the override blend at the rewound time
        app.taskMgr.step()
        if wrapped:
            clock.game_day += 1
    clock.game_time_of_day = target_s


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def capture(
    frames: int,
    out_name: str,
    explode: bool,
    time_of_day: float | None = None,
    weather: str | None = None,
    pitch_deg: float = -35.0,
    yaw_deg: float = 0.0,
    height_m: float | None = None,
    stub_sky: bool = False,
    torch: bool = False,
    game_day: int | None = None,
    flashlight: bool = False,
    gi_room: bool = False,
    move_into_room: bool = False,
    occluder: bool = False,
    no_grass: bool = False,
) -> Path:
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
        (renderer-only debugging; bypasses fire_engine.world.sky entirely).

    Returns
    -------
    Path
        The written PNG path.
    """
    import main as demo
    from fire_engine.core.math3d import Quat, Vec3
    from fire_engine.world.terrain import BrushMode, SphereBrush, apply_brush

    app = demo.build_demo()

    if game_day is not None:
        # Set BEFORE the warmup so moon phase / weather schedule match.
        app._clock.game_day = int(game_day)

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
        from fire_engine.render.sky.sky_renderer import SkyRendererComponent

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
        # Carve a crater at the terrain surface ahead of the camera so the
        # relit interior shows.  Raycast down for the surface — a fixed
        # camera-relative offset misses it entirely at some spawn heights
        # (the sphere ends up fully in air or fully buried → no visible edit).
        from fire_engine.world.terrain import raycast_voxel

        cam = app.camera_go.transform.position
        fwd = app.camera_go.transform.forward
        # 10 m ahead along the camera's horizontal facing (not a fixed axis).
        fl = max((fwd.x**2 + fwd.y**2) ** 0.5, 1e-6)
        ax, ay = cam.x + 10.0 * fwd.x / fl, cam.y + 10.0 * fwd.y / fl
        hit = raycast_voxel(
            Vec3(ax, ay, cam.z + 60.0),
            Vec3(0.0, 0.0, -1.0),
            app.chunk_manager.get_or_create,
            max_distance_m=150.0,
        )
        center = hit.point if hit is not None else Vec3(ax, ay, cam.z - 8.0)
        touched = apply_brush(
            SphereBrush(3.0),
            center,
            BrushMode.REMOVE,
            material=1,
            chunk_provider=app.chunk_manager.get_or_create,
            bus=app._event_bus,
        )
        # Same-frame remesh, matching the game's fire_explosion path.
        app.chunk_manager.remesh_edited(touched, app.light_sampler)
        print("EXPLODE at", center, "-", len(touched), "chunk(s) touched")

    if gi_room:
        # Build the Cornell-style GI test room ahead of the camera; optionally
        # step inside it (camera at the doorway looking at the far wall).
        cx, cy, z0 = demo.build_gi_test_room(app)
        print("GI ROOM at", (cx, cy, z0))
        if move_into_room:
            # Stand just inside the doorway and look down the room, tilted
            # slightly down so the floor, both coloured side walls and the
            # white back wall are framed — the red/green bounce onto the
            # white surfaces (and floor) is the GI signature to look for.
            app.camera_go.transform.position = Vec3(cx, cy - 4.2, z0 + 2.4)
            app.camera_go.transform.local_rotation = (
                Quat.from_axis_angle(Vec3.RIGHT, math.radians(-7.0))
            ).normalized()

    if flashlight and getattr(app, "lighting_pipeline", None) is not None:
        # Camera-mounted spot light (the F-key flashlight, statically placed).
        from fire_engine.lighting.lights import SpotLight

        cam = app.camera_go.transform.position
        fwd = app.camera_go.transform.forward
        app.lighting_pipeline.lights.add(
            SpotLight(
                position=(cam.x, cam.y, cam.z),
                direction=(fwd.x, fwd.y, fwd.z),
                color=(1.0, 0.96, 0.86),
                intensity=20.0,
                radius=36.0,
                cone_deg=38.0,
            )
        )
        print("FLASHLIGHT on at", (cam.x, cam.y, cam.z))

    if torch and getattr(app, "lighting_pipeline", None) is not None:
        # Drop a warm torch point-light ahead of the camera (GPU backend
        # only) so GI gather / volumetric glow can be captured headless.
        # Raycast for the surface — a fixed camera-relative offset buries the
        # torch below ground at some spawn heights (same failure the
        # --explode path had), which reads as "point lights are broken".
        from fire_engine.lighting.lights import PointLight
        from fire_engine.world.terrain import raycast_voxel as _rc_t

        cam = app.camera_go.transform.position
        t_hit = _rc_t(
            Vec3(cam.x, cam.y + 8.0, cam.z + 40.0),
            Vec3(0.0, 0.0, -1.0),
            app.chunk_manager.get_or_create,
            max_distance_m=90.0,
        )
        tz = (t_hit.point.z if t_hit is not None else cam.z - 2.5) + 2.0
        pos = (cam.x, cam.y + 8.0, tz)  # ~2 m above the surface
        app.lighting_pipeline.lights.add(
            PointLight(position=pos, color=(1.0, 0.62, 0.28), intensity=8.0, radius=16.0)
        )
        print("TORCH dropped at", pos)

    if occluder and getattr(app, "dev_overlay", None) is not None:
        # Spawn a real dev cube ~6 m ahead and drop it just above the ground.
        # The overlay's per-frame _sync_spawned pushes the cube's world AABB to
        # the lighting pipeline as a dynamic occluder, so the sun's boxVis test
        # in inject.comp must carve a shadow beneath it — the exact path the
        # user's "every object casts shadows" request exercises.
        from fire_engine.world.terrain import raycast_voxel as _rc

        cam = app.camera_go.transform.position
        fwd = app.camera_go.transform.forward
        gx, gy = cam.x + fwd.x * 6.0, cam.y + fwd.y * 6.0
        hit = _rc(
            Vec3(gx, gy, cam.z + 40.0),
            Vec3(0.0, 0.0, -1.0),
            app.chunk_manager.get_or_create,
            max_distance_m=90.0,
        )
        gz = hit.point.z if hit is not None else 8.0
        go = app.dev_overlay.spawn_cube()
        go.transform.position = Vec3(gx, gy, gz + 1.5)
        go.transform.local_scale = Vec3(3.0, 3.0, 3.0)  # 3 m cube, big shadow
        print("OCCLUDER dev cube at", (gx, gy, gz + 1.5))
        import os as _os

        if _os.environ.get("OCC_EMISSIVE"):
            # Mark the cube emissive: registers an AreaLight on its AABB so it
            # glows and lights its surroundings (the dynamic emission path).
            app.dev_overlay.toggle_emissive()
            print("OCCLUDER cube marked emissive")

    # Step the task manager so chunks stream, remesh, relight, and the sky
    # settles.  Each step runs the frame task AND flips the window, so the
    # framebuffer is valid.  Hold the game clock at the requested time so long
    # warmups don't drift the sun (~3 game-minutes per real second otherwise).
    hold_tod = float(app._clock.game_time_of_day)
    for _ in range(frames):
        app.taskMgr.step()
        if time_of_day is not None:
            app._clock.game_time_of_day = hold_tod

    if gi_room or no_grass or occluder:
        # Grass clutters the GI box and adds per-frame sway noise that swamps
        # A/B shadow diffs.  The grass root only exists after the first frame,
        # so hide it now and render one more frame before capturing.
        grass_go = getattr(app, "grass_go", None)
        if grass_go is not None:
            from fire_engine.render.vegetation.grass_renderer import GrassRendererComponent

            gc = grass_go.get_component(GrassRendererComponent)
            if gc is not None and getattr(gc, "_root", None) is not None:
                gc._root.hide()
                app.taskMgr.step()
                if time_of_day is not None:
                    app._clock.game_time_of_day = hold_tod

    out_dir = _REPO_ROOT / "tools" / "out"
    out_path = out_dir / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Capture the framebuffer into a PNMImage and write it (more reliable than
    # win.save_screenshot, which can no-op if called between flips).
    from panda3d.core import Filename, PNMImage

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
    parser.add_argument(
        "--frames", type=int, default=180, help="frames to step before capture (default 180)"
    )
    parser.add_argument(
        "--out", default="demo.png", help="output PNG name under tools/out/ (subdirs ok)"
    )
    parser.add_argument("--explode", action="store_true", help="carve a crater before capturing")
    parser.add_argument(
        "--time-of-day",
        type=float,
        default=None,
        metavar="HOURS",
        help="game-clock hour 0-24 (e.g. 6.5 = 06:30)",
    )
    parser.add_argument(
        "--weather",
        default=None,
        choices=["clear", "cloudy", "overcast", "fog", "rain", "storm"],
        help="force a weather type (fully blended at capture)",
    )
    parser.add_argument(
        "--pitch",
        type=float,
        default=-35.0,
        help="camera pitch degrees (default -35; use +5..+30 for sky shots)",
    )
    parser.add_argument(
        "--yaw", type=float, default=0.0, help="camera yaw degrees (0 = +Y/north, -90 = +X/east)"
    )
    parser.add_argument(
        "--height",
        type=float,
        default=None,
        help="camera height in meters (e.g. 110 = above clouds)",
    )
    parser.add_argument(
        "--stub-sky",
        action="store_true",
        help="swap a SkyState stub into the sky renderer (renderer-only debugging)",
    )
    parser.add_argument(
        "--torch",
        action="store_true",
        help="drop a warm torch point-light ahead of the camera (GPU lighting backend only)",
    )
    parser.add_argument(
        "--day", type=int, default=None, help="game day number (moon phase: day 15 = full)"
    )
    parser.add_argument(
        "--flashlight",
        action="store_true",
        help="add a camera-mounted spot light (the F-key flashlight; GPU lighting backend only)",
    )
    parser.add_argument(
        "--gi-room",
        action="store_true",
        help="build the Cornell-style GI test room ahead of the camera before capturing",
    )
    parser.add_argument(
        "--inside",
        action="store_true",
        help="with --gi-room: move the camera inside the room (doorway view of the far wall)",
    )
    parser.add_argument(
        "--occluder",
        action="store_true",
        help="float a box occluder ahead of the camera to verify dynamic-object (dev-cube) shadows",
    )
    parser.add_argument(
        "--no-grass",
        action="store_true",
        help="hide the GPU grass before capture (clean A/B shadow diffs without sway noise)",
    )
    args = parser.parse_args()

    path = capture(
        args.frames,
        args.out,
        args.explode,
        time_of_day=args.time_of_day,
        weather=args.weather,
        pitch_deg=args.pitch,
        yaw_deg=args.yaw,
        height_m=args.height,
        stub_sky=args.stub_sky,
        torch=args.torch,
        game_day=args.day,
        flashlight=args.flashlight,
        gi_room=args.gi_room,
        move_into_room=args.inside,
        occluder=args.occluder,
        no_grass=args.no_grass,
    )
    # Report on stdout for CI.
    size = os.path.getsize(path) if path.exists() else 0
    print(f"SCREENSHOT_RESULT wrote {path} ({size} bytes)")
    sys.stdout.flush()
    # Force-exit so the lingering Panda3D window / OpenAL device don't hang us.
    os._exit(0)


if __name__ == "__main__":
    main()
