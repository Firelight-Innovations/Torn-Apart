# Session 02 — Procedural Sky + Weather

*Date: 2026-06-09 · Scope: owner request — a fully procedural skybox (raymarched boxy clouds, procedural sun, procedural galaxy + stars, shooting stars) plus a small weather system (rain, fog, storms) living as a GameObject/Component in the World API, with the lighting responding to time of day and weather.*

## What shipped

`python main.py` now boots at 10:00 game time under a living sky: a painterly Minecraft × Morrowind
atmosphere gradient, a procedural sun with disc + halo that rises 06:00 in the east and sets 18:00
in the west, Minecraft-style **boxy clouds raymarched in the fragment shader** that drift with the
wind, and — at night — a **procedural galaxy** (noise filaments, dust lanes, warm core) with ~2500
twinkling stars, a phase-correct moon, and deterministic **shooting stars**. A seeded Markov
**weather system** cycles CLEAR / CLOUDY / OVERCAST / FOG / RAIN / STORM through the day: rain falls
as parallax streak layers around the camera, fog swallows distant terrain, and the baked voxel
sunlight dims and tints with daylight and weather. Everything is a pure function of `world_seed` +
the game clock; the weather save delta is `{}` unless a dev override is active.

**Controls added:** F6 cycle forced weather (…→ natural), F7 time-scale 60↔1800, F8 jump +6 game-hours.
Verification renders: `tools/out/sky/*.png` (noon/dawn/midnight/rain/fog/overcast/above-the-clouds).

## Architecture (two halves, one frozen contract)

```
fire_engine/sky/            headless, panda3d-free (Layer 1 — Services, peer of lighting/)
  celestial.py             sun/moon great-circle arcs from clock.game_time_of_day, daylight factor
  weather.py               WeatherType, WeatherParams, WeatherSystem (Saveable "weather",
                           2-game-hour Markov segments via for_domain("weather", day, segment),
                           20-game-minute blends, force_weather override, WeatherChangedEvent)
  sky_state.py             SkyState (frozen per-frame snapshot) + SkySystem aggregator

fire_engine/world/          the only panda3d zone
  sky_shaders.py           GLSL: dome (gradient, sun, moon, equirect night sky + twinkle,
                           shooting-star streak) + clouds (2-D DDA slab raymarch, ≤48 steps)
  sky_renderer.py          SkyRendererComponent on the "Sky" GameObject — builds dome/cloud
                           quads/rain cylinders once (bulk numpy→memoryview), then per frame:
                           sky_system.update() in update(), ~30 uniform writes in late_update(),
                           fog + clear color + terrain_root.set_color_scale(*terrain_light_scale)

procedural textures        "night_sky" (1024×512 equirect galaxy + stars, alpha = luminance)
                           "rain_streak" (128×512 tileable streaks)
```

The renderer reads **only** the `SkyState` dataclass — the halves were built concurrently by two
agents against this contract and met cleanly. Lighting integration is deliberately v0: the baked
vertex sunlight is modulated by one global `set_color_scale` per frame (see DECISIONS.md, "Day/night
+ weather lighting integration is a global colour-scale").

## Determinism

Same seed → identical weather schedule (pure function of `(world_seed, day, segment)`, day-anchored),
identical `night_sky`/`rain_streak` texture bytes, identical shooting-star schedule
(`for_domain("sky", "shooting_stars", day, slot)`), identical cloud field (seed uniform from
`for_domain("sky", "clouds")`). Weather saves are ~0 bytes (delta only on `force_weather`).

## Tests

`pytest -q`: **353 passed, 1 deselected** (was 307 before this session). New: 24 sky (sun geometry,
continuity, SkyState determinism, light-scale bounds), 13 weather (schedule determinism, blend
continuity, save round-trip, event publication), 9 procedural (both textures: shape/dtype,
byte-identical regen, seed divergence).

## Bugs found & fixed during the session

1. **Rain fell upward with an upside-down streak texture** (owner-reported live). Root cause:
   `texture_bridge.to_panda_texture` vertically flips every texture for OpenGL; the cylinder V axis
   and scroll sign didn't account for it. Fixed render-side (mirrored V + corrected scroll); verified
   with two frames 0.08 s apart (heads lead down, pattern translates down). `world.md` gotcha 11,
   `sky.md` gotcha 9.
2. **Coverage-threshold clouds vanished below 0.3** — raw thresholding of bell-distributed noise;
   fixed with a CPU quantile table so `cloud_coverage` is the true fill fraction (gotcha 12).
3. **Cell-seam slivers on distant cloud ceilings** — fixed with entry-face carry + 0.05 m interval
   overlap in the DDA (gotcha 13).

## Known limitations / deferred

- **Lighting is modulated, not re-lit:** shadows don't track the sun's angle (the column pass is
  vertical-only); clouds don't shadow the ground; point lights still don't exist. A sun-angle-aware
  pass or GPU relight is the upgrade path.
- **Rain is cosmetic** — no splashes, no gameplay effect, no sound (audio system doesn't exist yet).
- **Moon phase shades the disc but doesn't change moonlight intensity.**
- **Weather has no biome/seasonal variation** (single global Markov table; seasons exist on the
  Clock but aren't consumed yet).
- The fixed 06:00/18:00 sun schedule ignores season/latitude (v0 by design).

## How it was built

Two subagents in parallel against a frozen `SkyState`/`WeatherSystem` contract: one owned the
headless package + textures + tests, the other owned shaders/renderer/wiring + screenshot-driven
visual iteration (5 capture rounds). The orchestrator reconciled docs, logged DECISIONS, and
verified the suite + renders. Doc updates shipped with the code: `docs/systems/sky.md` (new),
`core.md`, `procedural.md`, `world.md`, README controls.
