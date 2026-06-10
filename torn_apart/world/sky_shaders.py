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

Uniforms (set per frame unless noted):
    p3d_Texture0      sampler2D — "night_sky" 1024x512 equirect RGBA galaxy;
                      alpha channel marks per-pixel luminance (twinkle mask).
    u_sun_dir         vec3  — unit, toward the sun.
    u_sun_color       vec3  — RGB 0-1 sun tint (warm at horizon).
    u_sun_intensity   float — 0-1; scales disc + halo + glow.
    u_moon_dir        vec3  — unit, toward the moon.
    u_moon_phase      float — 0-1; 0.5 = full moon (terminator shading).
    u_zenith_color    vec3  — RGB 0-1 gradient top.
    u_horizon_color   vec3  — RGB 0-1 gradient bottom.
    u_star_visibility float — 0-1; scales the night-sky texture + twinkle.
    u_star_rotation   float — radians; whole-sky rotation about world +Z
                      (slow sidereal drift over the night).
    u_time            float — real seconds since component start (twinkle hash).
    u_fog_color       vec3  — RGB 0-1; horizon band blends toward this.
    u_fog_blend       float — 0-1; how strongly fog swallows the horizon
                      (derived from SkyState.fog_density on the CPU).
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
uniform sampler2D p3d_Texture0;     // night_sky equirect (alpha = luminance)

uniform vec3  u_sun_dir;
uniform vec3  u_sun_color;
uniform float u_sun_intensity;
uniform vec3  u_moon_dir;
uniform float u_moon_phase;
uniform vec3  u_zenith_color;
uniform vec3  u_horizon_color;
uniform float u_star_visibility;
uniform float u_star_rotation;
uniform float u_time;
uniform vec3  u_fog_color;
uniform float u_fog_blend;
uniform float u_ss_active;
uniform vec3  u_ss_start;
uniform vec3  u_ss_travel;
uniform float u_ss_progress;

in vec3 v_dir;
out vec4 frag_color;

const float PI = 3.14159265358979;

// Stable 3D -> 1D hash (no trig, no texture) for star twinkle.
float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

void main() {
    vec3 d = normalize(v_dir);

    // --- Atmosphere gradient: horizon -> zenith with a non-linear curve ----
    float elev = clamp(d.z, 0.0, 1.0);
    float grad = pow(elev, 0.42);                       // painterly bias to horizon band
    vec3 col = mix(u_horizon_color, u_zenith_color, grad);
    // Below the horizon: keep the horizon colour, gently darkening downward.
    if (d.z < 0.0) {
        col = u_horizon_color * (1.0 + d.z * 0.55);
    }
    // Slight desaturation for the painterly Morrowind feel.
    float luma = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(col, vec3(luma), 0.07);

    // --- Night sky: equirect galaxy + stars, slow rotation about +Z --------
    float ca = cos(u_star_rotation);
    float sa = sin(u_star_rotation);
    vec3 sd = vec3(ca * d.x - sa * d.y, sa * d.x + ca * d.y, d.z);
    vec2 sky_uv = vec2(atan(sd.y, sd.x) / (2.0 * PI) + 0.5,
                       asin(clamp(sd.z, -1.0, 1.0)) / PI + 0.5);
    vec4 night = texture(p3d_Texture0, sky_uv);
    // Twinkle: hash a quantised view direction against coarse time; only the
    // bright pixels (alpha = luminance mask) flicker.
    float tw = hash13(floor(sd * 220.0) + floor(u_time * 7.0));
    float twinkle = mix(1.0, 0.35 + 1.15 * tw, smoothstep(0.45, 0.9, night.a));
    vec3 stars = night.rgb * twinkle * u_star_visibility;
    stars *= smoothstep(-0.06, 0.18, d.z);              // sink into horizon haze
    col += stars;

    // --- Shooting star: bright fading streak along a great circle ----------
    if (u_ss_active > 0.5) {
        vec3 s = u_ss_start;                            // unit (CPU-normalised)
        vec3 tv = u_ss_travel;                          // unit tangent (CPU-orthogonalised)
        vec3 n = cross(s, tv);                          // path plane normal
        float dist_plane = dot(d, n);                   // ~ angular distance from path
        float along = atan(dot(d, tv), dot(d, s));      // radians along the path
        float arc = 0.55;                               // total streak travel (rad)
        float head = u_ss_progress * arc;
        float behind = head - along;                    // >0 -> in the tail
        float tail_lum = (behind >= 0.0) ? exp(-behind * 30.0) : 0.0;
        float width = exp(-(dist_plane * dist_plane) / (2.0 * 0.003 * 0.003));
        float fade = sin(PI * clamp(u_ss_progress, 0.0, 1.0));
        col += vec3(1.0, 0.97, 0.88) * tail_lum * width * fade * 1.8 * u_star_visibility;
    }

    // --- Moon: pale disc with phase terminator + faint halo ----------------
    vec3 md = u_moon_dir;
    float mc = dot(d, md);
    // disc angular radius ~0.40 deg: edge between cos(0.45 deg) and cos(0.35 deg)
    float mdisc = smoothstep(0.9999692, 0.9999813, mc);
    if (mdisc > 0.0) {
        vec3 ref = (abs(md.z) > 0.97) ? vec3(0.0, 1.0, 0.0) : vec3(0.0, 0.0, 1.0);
        vec3 mt = normalize(cross(ref, md));
        vec3 mb = cross(md, mt);
        const float moon_ang_r = 0.0070;                // rad
        vec2 ml = vec2(dot(d, mt), dot(d, mb)) / moon_ang_r;   // disc-local [-1,1]
        float r2 = clamp(dot(ml, ml), 0.0, 1.0);
        float mz = sqrt(1.0 - r2);                      // sphere bulge toward viewer
        float ph = (u_moon_phase - 0.5) * 2.0 * PI;     // 0 at full, +/-PI at new
        vec3 mlight = vec3(sin(ph), 0.0, cos(ph));      // disc-space light dir
        float lit = smoothstep(-0.08, 0.28, dot(vec3(ml, mz), mlight));
        vec3 moon_col = vec3(0.83, 0.85, 0.90) * (0.10 + 0.92 * lit);
        col = mix(col, moon_col, mdisc);
    }
    col += vec3(0.75, 0.78, 0.85) * pow(max(mc, 0.0), 1600.0) * 0.16;

    // --- Sun: crisp disc + soft halo + broad atmospheric glow --------------
    float sc = dot(d, u_sun_dir);
    // disc ~0.5 deg radius: edge between cos(0.55 deg) and cos(0.45 deg)
    float disc = smoothstep(0.9999539, 0.9999692, sc);
    float halo = pow(max(sc, 0.0), 900.0) * 0.55;
    float glow = pow(max(sc, 0.0), 6.0) * 0.20;
    col += u_sun_color * u_sun_intensity * (disc * 1.65 + halo + glow);

    // --- Fog swallows the horizon band --------------------------------------
    float fog_band = 1.0 - smoothstep(0.0, 0.38, d.z);
    col = mix(col, u_fog_color, u_fog_blend * fog_band);

    frag_color = vec4(col, 1.0);
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
