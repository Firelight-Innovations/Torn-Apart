"""
world/sky_shaders.py — GLSL sources for the procedural sky + cloud renderer.

Pure string constants (NO panda3d imports — this module is importable headless;
only ``world/sky_renderer.py`` compiles them via ``panda3d.core.Shader.make``).
All shaders are ``#version 330 core`` and use Panda3D's GLSL conventions:
``p3d_Vertex``, ``p3d_ModelViewProjectionMatrix``, ``p3d_ModelMatrix``,
``p3d_Texture0``; custom uniforms are fed via ``NodePath.set_shader_input``.

Coordinate conventions: world space, **Z-up**, meters.  All direction vectors
are unit length and point *toward* the body (sun_dir points at the sun).

SKY DOME (``SKY_DOME_VERTEX`` / ``SKY_DOME_FRAGMENT``)
-------------------------------------------------------
Painted on an inverted UV-sphere centred on the camera (translation-only
follow, so model-space vertex position IS the world view direction).

The daytime sky is a **per-pixel physical single-scattering raymarch**
(Rayleigh + Mie, constants mirrored verbatim from ``sky/atmosphere.py``):
blue zenith, bright horizon, the sunset/sunrise glow concentrated around
the sun azimuth, and the earth-shadow twilight arch all emerge from the
physics — there is no hand-painted gradient anymore.  The sun is a large
(~1.25°) limb-darkened disc tinted by its own atmospheric transmittance;
the moon is a large (~1.0°) disc textured with the procedural
``"moon_surface"`` texture and shaded by the dynamic phase terminator.
HDR output is ACES-tonemapped with the same exposure as the terrain shader;
the LDR night-sky art (stars/galaxy/twinkle/shooting stars) composites
after tonemapping so it stays crisp.

Uniforms (set per frame unless noted):
    u_star_cube       samplerCube — "night_sky_cube" 6×512² galaxy + star
                      faces (set once); alpha = per-pixel luminance (twinkle
                      mask).  Cube sampling kills the old equirect pole
                      pinch/distortion.
    u_celestial_axis  vec3  — unit, toward the celestial north pole (set
                      once; tilted by the world's seed-derived latitude so
                      stars rise/set instead of pinwheeling around zenith).
    u_moon_tex        sampler2D — "moon_surface" 256x256 lunar disc (set once).
    u_sun_dir         vec3  — unit, toward the sun.
    u_sun_color       vec3  — legacy tint (disc color now derives from the
                      atmosphere transmittance; kept for stub skies).
    u_sun_intensity   float — 0-1; scales disc + halo (cloud-dimmed).
    u_moon_dir        vec3  — unit, toward the moon.
    u_moon_phase      float — 0-1; 0.5 = full moon (terminator shading).
    u_moon_glow       float — 0-1 illuminated fraction; gates the moonlit-sky
                      scatter and the moon halo (new moon = dark sky).
    u_zenith_color    vec3  — weather-graded gradient top (legacy consumers).
    u_horizon_color   vec3  — weather-graded gradient bottom.
    u_star_visibility float — 0-1; scales the night-sky texture + twinkle.
    u_star_rotation   float — radians; whole-celestial-sphere rotation about
                      ``u_celestial_axis`` (one revolution per game day).
    u_time            float — real seconds since component start (twinkle hash).
    u_daylight        float — SkyState.daylight; blends in the night floor.
    u_weather_gray    float — 0-1 overcast desaturation weight.
    u_exposure        float — ACES tonemap exposure (matches terrain shader).
    u_fog_color       vec3  — RGB 0-1; horizon band blends toward this.
    u_fog_blend       float — 0-1 legacy horizon fog (CPU lighting backend;
                      forced to 0 under external/GPU lighting).
    u_fog_integrated  sampler3D — froxel fog accumulation (GPU backend only).
    u_fog_enabled     float — 0/1; gates the froxel fog composite.
    u_viewport        vec2  — window pixel size (froxel screen UV).
    u_ss_active       float — 0 or 1; shooting-star streak enable.
    u_ss_start        vec3  — unit view dir of the streak's spawn point (set
                      once per spawn).
    u_ss_travel       vec3  — unit travel direction (tangent; orthogonalised
                      against u_ss_start on the CPU).
    u_ss_progress     float — 0-1 animation progress along the great circle.

BOXY CLOUDS (``CLOUD_VERTEX`` / ``CLOUD_FRAGMENT``)
----------------------------------------------------
Raymarched Minecraft-style box clouds: a 2D DDA walks a grid of
``u_cell``-sized cells through the slab ``[u_altitude, u_altitude+u_thickness]``;
occupied cells are crisp axis-aligned boxes with per-cell height variation.
Drawn on two camera-following horizontal quads (slab bottom + slab top) so
fragments exist whether the camera is below, inside, or above the layer;
duplicate plane coverage is discarded in the shader.

Uniforms:
    u_cam_pos       vec3  — camera world position, meters (per frame).
    u_altitude      float — slab bottom Z, meters (set once; config
                    sky_cloud_altitude_m).
    u_thickness     float — slab thickness, meters (set once; config
                    sky_cloud_thickness_m).
    u_cell          float — cell edge, meters (set once; config sky_cloud_cell_m).
    u_seed          float — world-seed-derived hash offset (set once; from
                    core.rng.for_domain("sky", "clouds")).
    u_coverage      float — 0-1 fill fraction threshold (SkyState.cloud_coverage).
    u_opacity       float — 0-1 overall alpha scale (from SkyState.cloud_density).
    u_wind_offset   vec2  — accumulated wind drift, meters (CPU integrates
                    wind_dir * wind_speed * dt each frame).
    u_top_color     vec3  — flat-face colour for box tops (sunlit, computed CPU-side).
    u_side_color    vec3  — flat-face colour for box sides.
    u_bottom_color  vec3  — flat-face colour for box bottoms (darkest; storm-gray
                    when density is high).
    u_fade_dist     float — meters; clouds fade to transparent approaching this.

The fragment colour is emitted non-premultiplied for standard M_alpha blending.
"""

from __future__ import annotations

__all__ = [
    "SKY_DOME_VERTEX",
    "SKY_DOME_FRAGMENT",
    "CLOUD_VERTEX",
    "CLOUD_FRAGMENT",
]


# ---------------------------------------------------------------------------
# Sky dome
# ---------------------------------------------------------------------------

SKY_DOME_VERTEX: str = """
#version 330 core
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
out vec3 v_dir;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    // The dome is camera-centred with no rotation, so the model-space vertex
    // position is exactly the world-space view direction.
    v_dir = p3d_Vertex.xyz;
}
"""


SKY_DOME_FRAGMENT: str = """
#version 330 core
uniform samplerCube u_star_cube;    // night_sky_cube faces (alpha = luminance)
uniform sampler2D u_moon_tex;       // procedural "moon_surface" lunar disc
uniform vec3  u_celestial_axis;     // unit, toward the celestial north pole

uniform vec3  u_sun_dir;
uniform vec3  u_sun_color;          // legacy tint (kept for stubs); disc uses transmittance
uniform float u_sun_intensity;
uniform vec3  u_moon_dir;
uniform float u_moon_phase;
uniform float u_moon_glow;          // 0-1 illuminated fraction (moonlit-sky gain)
uniform vec3  u_zenith_color;       // legacy/weather-graded (below-horizon fill)
uniform vec3  u_horizon_color;
uniform float u_star_visibility;
uniform float u_star_rotation;
uniform float u_time;
uniform float u_daylight;           // SkyState.daylight (night-floor blend)
uniform float u_weather_gray;       // 0-1 overcast desaturation weight
uniform float u_exposure;           // tonemap exposure (match terrain shader)
uniform vec3  u_fog_color;
uniform float u_fog_blend;          // legacy horizon fog (CPU backend; 0 on GPU)
uniform sampler3D u_fog_integrated; // froxel fog (GPU backend)
uniform float u_fog_enabled;
uniform vec2  u_viewport;
uniform float u_ss_active;
uniform vec3  u_ss_start;
uniform vec3  u_ss_travel;
uniform float u_ss_progress;

in vec3 v_dir;
out vec4 frag_color;

const float PI = 3.14159265358979;

// --- Physical atmosphere constants (mirror sky/atmosphere.py exactly) -----
const vec3  BETA_R  = vec3(5.8e-6, 13.5e-6, 33.1e-6);
const float BETA_M  = 3.9e-6;
const float MIE_EXT = 1.1;
const float HR      = 8500.0;
const float HM      = 1200.0;
const float RP      = 6371000.0;
const float RT      = 6431000.0;          // RP + 60 km
const float R0      = 6371002.0;          // RP + observer 2 m
const float SUN_I   = 22.0;               // SUN_TOA_RADIANCE
const float MIE_G   = 0.76;
const int   STEPS   = 12;
const int   LSTEPS  = 4;

// Disc angular radii — ~2.5x their realistic sizes, per art direction.
const float SUN_ANG_R  = 0.0218;          // ~1.25 deg
const float MOON_ANG_R = 0.0175;          // ~1.00 deg

// Stable 3D -> 1D hash (no trig, no texture) for star twinkle.
float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

float exitDist(float r, float cb) {
    float disc = cb * cb - (r * r - RT * RT);
    return -cb + sqrt(max(disc, 0.0));
}

// Transmittance from the observer toward dir (sun/moon disc tint).
vec3 viewTransmittance(vec3 d) {
    float cb = R0 * max(d.z, 0.0);
    float tExit = exitDist(R0, cb);
    float odR = 0.0;
    float odM = 0.0;
    float dt = tExit / 8.0;
    for (int i = 0; i < 8; i++) {
        float t = (float(i) + 0.5) * dt;
        float r = sqrt(R0 * R0 + t * t + 2.0 * t * cb);
        float h = max(r - RP, 0.0);
        odR += exp(-h / HR) * dt;
        odM += exp(-h / HM) * dt;
    }
    return exp(-(BETA_R * odR + BETA_M * MIE_EXT * odM));
}

// Single-scattered radiance along view ray d for a celestial light at s —
// the same integral as sky/atmosphere.py sky_radiance(), fewer steps.
vec3 scatterLight(vec3 d, vec3 s, float lightI) {
    float cb = R0 * d.z;
    float tExit = exitDist(R0, cb);
    float discP = cb * cb - (R0 * R0 - RP * RP);
    if (cb < 0.0 && discP > 0.0) tExit = min(tExit, -cb - sqrt(discP));
    float mu = dot(d, s);
    float phR = (3.0 / (16.0 * PI)) * (1.0 + mu * mu);
    float g2 = MIE_G * MIE_G;
    float phM = (3.0 / (8.0 * PI)) * (1.0 - g2) * (1.0 + mu * mu)
              / ((2.0 + g2) * pow(1.0 + g2 - 2.0 * MIE_G * mu, 1.5));
    float odRv = 0.0;
    float odMv = 0.0;
    vec3 L = vec3(0.0);
    for (int i = 0; i < STEPS; i++) {
        // Quadratic spacing (mirrors sky/atmosphere.py): dense sampling near
        // the observer so grazing rays see the low, dense atmosphere.
        float u = (float(i) + 0.5) / float(STEPS);
        float t = tExit * u * u;
        float dt = tExit * (2.0 * u / float(STEPS));
        float r = sqrt(R0 * R0 + t * t + 2.0 * t * cb);
        float h = max(r - RP, 0.0);
        float dR = exp(-h / HR);
        float dM = exp(-h / HM);
        odRv += dR * dt;
        odMv += dM * dt;
        // Sun-ray from the sample: planet-shadow test gives the earth-shadow
        // twilight arch after sunset.
        float cbl = R0 * s.z + t * mu;
        float discL = cbl * cbl - (r * r - RP * RP);
        if (cbl < 0.0 && discL > 0.0) continue;
        float tl = exitDist(r, cbl);
        float dl = tl / float(LSTEPS);
        float odRl = 0.0;
        float odMl = 0.0;
        for (int j = 0; j < LSTEPS; j++) {
            float t2 = (float(j) + 0.5) * dl;
            float rl = sqrt(r * r + t2 * t2 + 2.0 * t2 * cbl);
            float hl = max(rl - RP, 0.0);
            odRl += exp(-hl / HR) * dl;
            odMl += exp(-hl / HM) * dl;
        }
        vec3 tau = BETA_R * (odRv + odRl) + BETA_M * MIE_EXT * (odMv + odMl);
        L += exp(-tau) * (BETA_R * (dR * phR) + BETA_M * (dM * phM)) * dt;
    }
    return lightI * L;
}

// Disc-local coordinates for a celestial body; returns the disc mask.
float discFactor(vec3 d, vec3 dirTo, float angR, out vec2 local, out float rr) {
    vec3 ref = (abs(dirTo.z) > 0.97) ? vec3(0.0, 1.0, 0.0) : vec3(0.0, 0.0, 1.0);
    vec3 t = normalize(cross(ref, dirTo));
    vec3 b = cross(dirTo, t);
    local = vec2(dot(d, t), dot(d, b)) / angR;
    rr = length(local);
    return 1.0 - smoothstep(0.92, 1.0, rr);
}

vec3 acesTonemap(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14),
                 0.0, 1.0);
}

void main() {
    vec3 d = normalize(v_dir);

    // --- Physically-scattered sky (Rayleigh + Mie single scattering) -------
    // Below the geometric horizon (the dome shows past the finite world's
    // edge) the scatter ray is clamped to graze the horizon so the band
    // continues the horizon hue instead of going black at the ground hit.
    vec3 ds = (d.z < 0.015) ? normalize(vec3(d.xy, 0.015)) : d;
    vec3 col = scatterLight(ds, u_sun_dir, SUN_I);
    // Moonlit sky: same physics, tiny intensity, scaled by the phase.
    if (u_moon_dir.z > -0.05 && u_moon_glow > 0.01) {
        col += scatterLight(ds, u_moon_dir, SUN_I * 0.0045 * u_moon_glow);
    }
    // Overcast: desaturate toward gray (luminance-preserving).
    float luma = dot(col, vec3(0.2126, 0.7152, 0.0722));
    col = mix(col, vec3(luma), 0.75 * u_weather_gray);
    // Artistic night floor (matches SkySystem's _NIGHT_* constants).
    float grad = pow(clamp(d.z, 0.0, 1.0), 0.42);
    col += mix(vec3(0.020, 0.026, 0.046), vec3(0.012, 0.016, 0.035), grad)
           * (1.0 - u_daylight);
    // Below the horizon: gently dim the continued horizon band downward.
    if (d.z < 0.0) col *= clamp(1.0 + d.z * 1.2, 0.45, 1.0);

    vec3 sunT = viewTransmittance(u_sun_dir);

    // --- Sun: large limb-darkened disc tinted by its own transmittance -----
    float sc = dot(d, u_sun_dir);
    if (sc > 0.999) {
        vec2 sl;
        float sr;
        float sdisc = discFactor(d, u_sun_dir, SUN_ANG_R, sl, sr);
        if (sdisc > 0.0) {
            float limb = sqrt(max(1.0 - sr * sr, 0.0));
            float ld = 0.42 + 0.58 * limb;               // limb darkening
            col += sunT * (u_sun_intensity * sdisc * ld * 14.0);
        }
    }
    // Forward-Mie halo around the sun.
    col += sunT * (u_sun_intensity * pow(max(sc, 0.0), 350.0) * 0.55);

    // --- Moon: large textured disc with dynamic phase terminator -----------
    float mc = dot(d, u_moon_dir);
    if (mc > 0.999) {
        vec2 ml;
        float mr;
        float mdisc = discFactor(d, u_moon_dir, MOON_ANG_R, ml, mr);
        if (mdisc > 0.0) {
            vec3 mtex = texture(u_moon_tex, ml * 0.5 + 0.5).rgb;
            float mz = sqrt(max(1.0 - clamp(mr * mr, 0.0, 1.0), 0.0));
            float ph = (u_moon_phase - 0.5) * 2.0 * PI;   // 0 at full
            vec3 mlight = vec3(sin(ph), 0.0, cos(ph));
            float lit = smoothstep(-0.08, 0.28, dot(vec3(ml, mz), mlight));
            vec3 moonCol = mtex * viewTransmittance(u_moon_dir)
                         * (0.05 + 1.10 * lit) * 1.5;
            col = mix(col, moonCol, mdisc);               // moon occludes sky
        }
    }
    // Faint moon halo, night only.
    col += vec3(0.75, 0.78, 0.85)
         * (pow(max(mc, 0.0), 1600.0) * 0.10 * (1.0 - u_daylight) * u_moon_glow);

    // --- Volumetric fog composite (GPU backend: froxel far slice) ----------
    if (u_fog_enabled > 0.5) {
        vec4 fog = texture(u_fog_integrated,
                           vec3(gl_FragCoord.xy / u_viewport, 1.0));
        col = col * fog.a + fog.rgb;
    }
    // Legacy horizon fog band (CPU backend; u_fog_blend forced 0 on GPU).
    float fog_band = 1.0 - smoothstep(0.0, 0.38, d.z);
    col = mix(col, u_fog_color, u_fog_blend * fog_band);

    // --- Tonemap to match the terrain shader -------------------------------
    vec3 ldr = pow(acesTonemap(col * u_exposure), vec3(1.0 / 2.2));

    // --- Night sky art (stars/galaxy/twinkle) added in LDR, post-tonemap ---
    // The whole celestial sphere rotates about the TILTED celestial axis
    // (Polaris elevation = the world's latitude), not world +Z — so the
    // stars rise in the east and set in the west instead of pinwheeling
    // overhead.  Rodrigues rotation of the view dir into the star frame.
    vec3 ax = u_celestial_axis;
    float cr = cos(u_star_rotation);
    float sr = sin(u_star_rotation);
    vec3 sd = d * cr + cross(ax, d) * sr + ax * (dot(ax, d) * (1.0 - cr));
    vec4 night = texture(u_star_cube, sd);
    // Per-star flicker: a stable hash over the star-frame direction picks
    // each star's phase + speed; a smooth sine gives a lively shimmer
    // (galaxy band barely flickers — the alpha mask gates it to stars).
    float h = hash13(floor(sd * 300.0));
    float tw01 = 0.5 + 0.5 * sin(u_time * (3.0 + 6.0 * h) + h * 6.2831853);
    float twinkle = mix(1.0, 0.30 + 1.40 * tw01,
                        smoothstep(0.40, 0.85, night.a));
    vec3 stars = night.rgb * twinkle * u_star_visibility;
    stars *= smoothstep(-0.06, 0.18, d.z);              // sink into horizon haze
    ldr += stars;

    // --- Shooting star: bright fading streak along a great circle ----------
    if (u_ss_active > 0.5) {
        vec3 s = u_ss_start;
        vec3 tv = u_ss_travel;
        vec3 n = cross(s, tv);
        float dist_plane = dot(d, n);
        float along = atan(dot(d, tv), dot(d, s));
        float arc = 0.55;
        float head = u_ss_progress * arc;
        float behind = head - along;
        float tail_lum = (behind >= 0.0) ? exp(-behind * 30.0) : 0.0;
        float width = exp(-(dist_plane * dist_plane) / (2.0 * 0.003 * 0.003));
        float fade = sin(PI * clamp(u_ss_progress, 0.0, 1.0));
        ldr += vec3(1.0, 0.97, 0.88) * tail_lum * width * fade * 1.8
             * u_star_visibility;
    }

    frag_color = vec4(ldr, 1.0);
}
"""


# ---------------------------------------------------------------------------
# Boxy raymarched clouds
# ---------------------------------------------------------------------------

CLOUD_VERTEX: str = """
#version 330 core
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
in vec4 p3d_Vertex;
out vec3 v_world;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    v_world = (p3d_ModelMatrix * p3d_Vertex).xyz;
}
"""


CLOUD_FRAGMENT: str = """
#version 330 core
uniform vec3  u_cam_pos;
uniform float u_altitude;
uniform float u_thickness;
uniform float u_cell;
uniform float u_seed;
uniform float u_coverage;
uniform float u_opacity;
uniform vec2  u_wind_offset;
uniform vec3  u_top_color;
uniform vec3  u_side_color;
uniform vec3  u_bottom_color;
uniform float u_fade_dist;

in vec3 v_world;
out vec4 frag_color;

const int MAX_STEPS = 48;

// 2D -> 1D hash, seeded by the world-seed uniform.
float hash21(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031 + u_seed);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Bilinear value noise over the integer lattice (cells as sample points).
float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float e = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, e, f.x), f.y);
}

// Per-cell occupancy value in ~[0,1]: 2-octave value noise (coarse billow
// shape dominates so low coverage gives Minecraft-style CLUMPS, not lone
// cells) + a small per-cell hash for ragged edges.  Occupied when
// value < u_coverage, so coverage directly controls the fill fraction.
float cell_value(vec2 cell) {
    return 0.55 * vnoise(cell / 6.0) + 0.30 * vnoise(cell / 2.2)
         + 0.15 * hash21(cell + 17.0);
}

void main() {
    vec3 ro = u_cam_pos;
    vec3 rd = normalize(v_world - ro);

    float zb = u_altitude;
    float zt = u_altitude + u_thickness;

    // Two quads cover the slab (bottom plane + top plane).  A ray from below
    // the slab crosses BOTH planes -> would shade twice; discard the far one.
    bool on_bottom_plane = v_world.z < (zb + 0.5 * u_thickness);
    if (on_bottom_plane && ro.z > zt) discard;       // seen from above: keep top quad
    if (!on_bottom_plane && ro.z < zb) discard;      // seen from below: keep bottom quad

    // Ray / slab interval [t0, t1] in meters along the ray.
    float t0;
    float t1;
    if (abs(rd.z) < 1e-4) {
        if (ro.z < zb || ro.z > zt) discard;         // horizontal ray outside slab
        t0 = 0.0;
        t1 = u_fade_dist;
    } else {
        float ta = (zb - ro.z) / rd.z;
        float tb = (zt - ro.z) / rd.z;
        t0 = max(min(ta, tb), 0.0);
        t1 = min(max(ta, tb), u_fade_dist * 1.15);
    }
    if (t1 <= t0) discard;

    // 2D DDA over the cell grid (XY plane), wind-shifted.
    vec2 p0 = ro.xy + rd.xy * t0 + u_wind_offset;
    vec2 cell = floor(p0 / u_cell);
    float sx = (rd.x >= 0.0) ? 1.0 : -1.0;
    float sy = (rd.y >= 0.0) ? 1.0 : -1.0;
    float tdx = (abs(rd.x) > 1e-6) ? u_cell / abs(rd.x) : 1e30;
    float tdy = (abs(rd.y) > 1e-6) ? u_cell / abs(rd.y) : 1e30;
    float bx = (sx > 0.0) ? (cell.x + 1.0) * u_cell : cell.x * u_cell;
    float by = (sy > 0.0) ? (cell.y + 1.0) * u_cell : cell.y * u_cell;
    float tmx = (abs(rd.x) > 1e-6) ? t0 + (bx - p0.x) / rd.x : 1e30;
    float tmy = (abs(rd.y) > 1e-6) ? t0 + (by - p0.y) / rd.y : 1e30;

    float t = t0;
    float acc = 0.0;
    vec3 col = vec3(0.0);
    // Shading continuity across SHARED faces: when the ray leaves one
    // occupied box directly into the next, the "side-face entry" of the
    // second box is an interior face that should not be visible — without
    // this carry, every cell seam draws a bright side-coloured grid line
    // over distant cloud ceilings/floors.
    bool prev_hit = false;
    vec3 carry_col = vec3(0.0);

    for (int i = 0; i < MAX_STEPS; ++i) {
        float t_exit = min(min(tmx, tmy), t1);
        bool hit = false;

        if (cell_value(cell) < u_coverage) {
            // Crisp box: full cell footprint, per-cell top height variation
            // within the slab for a chunky skyline.
            float topz = zb + u_thickness * (0.45 + 0.55 * hash21(cell + 7.7));
            // Tiny interval overlap: float32 precision at glancing angles
            // otherwise loses a dark sliver at every cell seam.
            float bt0 = max(t - 0.05, t0);
            float bt1 = min(t_exit + 0.05, t1);
            if (abs(rd.z) > 1e-5) {
                float za = (zb - ro.z) / rd.z;
                float zc = (topz - ro.z) / rd.z;
                bt0 = max(bt0, min(za, zc));
                bt1 = min(bt1, max(za, zc));
            } else if (ro.z > topz) {
                bt1 = bt0 - 1.0;                     // horizontal ray above this box
            }
            if (bt1 > bt0) {
                hit = true;
                // Flat-face lighting: which face did the ray enter?
                float ez = ro.z + rd.z * bt0;
                vec3 face_col;
                if (ez >= topz - 0.06)      face_col = u_top_color;
                else if (ez <= zb + 0.06)   face_col = u_bottom_color;
                else                        face_col = prev_hit ? carry_col
                                                                : u_side_color;
                carry_col = face_col;

                float a = 1.0 - exp(-(bt1 - bt0) * 0.55);          // chunk opacity
                a *= 1.0 - smoothstep(u_fade_dist * 0.45, u_fade_dist, bt0);
                a *= u_opacity;

                col += (1.0 - acc) * a * face_col;
                acc += (1.0 - acc) * a;
                if (acc > 0.98) break;               // early-out: opaque enough
            }
        }
        prev_hit = hit;

        // Advance to the next cell.
        if (tmx < tmy) { cell.x += sx; t = tmx; tmx += tdx; }
        else           { cell.y += sy; t = tmy; tmy += tdy; }
        if (t >= t1) break;                          // beyond slab / fade distance
    }

    if (acc < 0.004) discard;
    frag_color = vec4(col / max(acc, 1e-4), acc);    // un-premultiply for M_alpha
}
"""
