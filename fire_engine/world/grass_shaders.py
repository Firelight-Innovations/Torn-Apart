"""
world/grass_shaders.py — GLSL for the GPU-only instanced grass.

The CPU never stores a blade: every instance derives its placement in the
vertex shader from ``gl_InstanceID`` via the lowbias32 hash chain that
``zones/grass_placement.py`` mirrors line-for-line (edit BOTH or the headless
placement tests lie about what the GPU draws).

Vertex shader
-------------
1. Hash ``gl_InstanceID`` (+ per-volume ``u_hash_seed``) → base XY inside
   ``u_bounds_min/max``, yaw, scale jitter, sway phase, tint.
2. Sample ``u_height_field`` (R channel: terrain surface height inside the
   volume's Z window; 255 = no ground) — sentinel or fully-faded instances
   collapse to a clip-space point and rasterise nothing (craters cull grass).
3. Sway: blade-local Z² weighted lean along ``u_wind_dir`` — a static lean
   (``u_sway_base``) plus a gust oscillation (``u_sway_gust`` ×
   sin(``u_time_s·u_gust_freq`` + phase)); both amplitudes are computed
   CPU-side from the weather (storms move grass more).
4. Distance fade: blades shrink to nothing between ``u_fade_start_m`` and
   ``u_fade_end_m`` from the camera — no popping, no far-field shimmer.

Fragment shader
---------------
Binary alpha cutout of the pixel-art ``grass_tuft`` texture (discard < 0.5 —
no sorting, depth-write stays on), lit by the SAME radiance-cascade volumes
as the terrain: direct sun/moon × voxel-marched visibility + flood-fill GI,
sampled at the blade base quantised to the ``u_quant_m`` light-pixel grid, so
grass shows the identical pixelated light patches, torch glow and crater
shadows as the ground it stands on.  Froxel fog composites with one tap, then
ACES + gamma — matching ``world/terrain_shader.py``.

All cascade/fog/celestial uniforms use the ``GpuLightingPipeline`` surface
contract names and are **inherited** from ``terrain_root`` (the pipeline
binds and refreshes them there each frame); only the grass-specific uniforms
are set by ``GrassRendererComponent``.
"""

from __future__ import annotations

__all__ = ["GRASS_VERTEX", "GRASS_FRAGMENT"]


GRASS_VERTEX = """#version 330 core
uniform mat4 p3d_ModelViewProjectionMatrix;

// --- per-volume (set by GrassRendererComponent) -------------------------
uniform vec3  u_bounds_min;       // volume AABB min corner (world m)
uniform vec3  u_bounds_max;
uniform int   u_hash_seed;        // per-volume seed (zones/grass_placement.py)
uniform sampler2D u_height_field; // R: surface height in z-window; 255=none

// --- grass tuning (config [grass]) ---------------------------------------
uniform float u_blade_height_m;
uniform float u_fade_start_m;
uniform float u_fade_end_m;

// --- weather sway (per frame from SkyState) ------------------------------
uniform vec2  u_wind_dir;         // unit XY, direction wind blows toward
uniform float u_sway_base;        // static lean at the tip (meters)
uniform float u_sway_gust;        // oscillating lean amplitude (meters)
uniform float u_gust_freq;        // oscillation rate (rad/s)
uniform float u_time_s;

// --- shared lighting contract (inherited from terrain_root) --------------
uniform vec3  u_cam_pos;

in vec4 p3d_Vertex;               // blade-local position (z up, base at 0)
in vec2 p3d_MultiTexCoord0;

out vec2  v_uv;
out vec3  v_base_world;           // blade base (lighting sample point)
out float v_tint;                 // per-instance albedo jitter

// lowbias32 (Chris Wellons) — LINE-FOR-LINE mirror of
// zones/grass_placement.py::hash_lowbias32.  Edit both or neither.
uint lowbias32(uint x) {
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

float u2f(uint h) { return float(h) * (1.0 / 4294967296.0); }

void main() {
    // Hash chain — mirror of zones/grass_placement.py::instance_attribs.
    uint i  = uint(gl_InstanceID);
    uint h0 = lowbias32(i  ^ uint(u_hash_seed));
    uint h1 = lowbias32(h0 ^ 0x9e3779b9u);
    uint h2 = lowbias32(h1 ^ 0x85ebca6bu);
    uint h3 = lowbias32(h2 ^ 0xc2b2ae35u);
    uint h4 = lowbias32(h3 ^ 0x27d4eb2fu);

    vec3 size = u_bounds_max - u_bounds_min;
    vec2 base_xy = u_bounds_min.xy + vec2(u2f(h0), u2f(h1)) * size.xy;

    // Terrain surface under this blade (baked field; 255 = no ground).
    vec2 field_uv = (base_xy - u_bounds_min.xy) / size.xy;
    float r = texture(u_height_field, field_uv).r;

    // Distance fade: shrink to zero between fade_start and fade_end.
    float fade = 1.0 - smoothstep(u_fade_start_m, u_fade_end_m,
                                  distance(base_xy, u_cam_pos.xy));

    if (r * 255.0 > 254.5 || fade <= 0.001) {
        // Culled: collapse the whole instance to one clip-space point
        // outside the frustum — zero-area triangles, no fragments.
        gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
        v_uv = vec2(0.0);
        v_base_world = vec3(0.0);
        v_tint = 1.0;
        return;
    }

    float base_z = u_bounds_min.z + (r * 255.0 / 254.0) * size.z;

    // Per-blade yaw + scale jitter (0.7-1.3x), shrunk by the distance fade.
    float rot = u2f(h2) * 6.2831853;
    float scale = (0.7 + 0.6 * u2f(h3)) * fade;
    float c = cos(rot), s = sin(rot);
    vec2 lp = vec2(c * p3d_Vertex.x - s * p3d_Vertex.y,
                   s * p3d_Vertex.x + c * p3d_Vertex.y);

    // Weather sway: quadratic in normalised blade height (base pinned,
    // tip moves), static lean + gust oscillation, along the wind.
    float hn = clamp(p3d_Vertex.z / u_blade_height_m, 0.0, 1.0);
    float phase = u2f(h4) * 6.2831853;
    float lean = u_sway_base * (0.6 + 0.8 * u2f(h2))
               + u_sway_gust * sin(u_time_s * u_gust_freq + phase);

    vec3 wp = vec3(base_xy + lp * scale,
                   base_z + p3d_Vertex.z * scale);
    wp.xy += u_wind_dir * (lean * hn * hn);

    gl_Position = p3d_ModelViewProjectionMatrix * vec4(wp, 1.0);
    v_uv = p3d_MultiTexCoord0;
    v_base_world = vec3(base_xy, base_z);
    v_tint = 0.85 + 0.30 * u2f(h4);
}
"""


GRASS_FRAGMENT = """#version 330 core
in vec2  v_uv;
in vec3  v_base_world;
in float v_tint;

out vec4 frag_color;

uniform sampler2D u_tuft;         // grass_tuft alpha-cutout texture

// --- radiance cascades (GpuLightingPipeline surface contract;
//     bound/refreshed on terrain_root and inherited here) ----------------
uniform sampler3D u_c0_radiance;
uniform sampler3D u_c0_vis;       // r sun, g moon, b sky visibility
uniform vec3  u_c0_origin_m;
uniform float u_c0_cell_m;
uniform float u_c0_cells;
uniform sampler3D u_c1_radiance;
uniform sampler3D u_c1_vis;
uniform vec3  u_c1_origin_m;
uniform float u_c1_cell_m;
uniform float u_c1_cells;

uniform vec3  u_sun_dir;
uniform vec3  u_sun_radiance;
uniform vec3  u_moon_dir;
uniform vec3  u_moon_radiance;
uniform vec3  u_sky_ambient;
uniform float u_quant_m;          // light-pixel size (matches terrain)
uniform float u_exposure;

// --- froxel fog (same one-tap composite as the terrain shader) -----------
uniform sampler3D u_fog_integrated;
uniform float u_fog_near;
uniform float u_fog_far;
uniform float u_fog_enabled;
uniform vec2  u_viewport;
uniform vec3  u_cam_pos;

vec3 c_uv(vec3 wp, vec3 origin, float cell_m, float cells) {
    return (wp - origin) / (cell_m * cells);
}

bool inBox(vec3 uv, float pad) {
    return all(greaterThan(uv, vec3(pad))) && all(lessThan(uv, vec3(1.0 - pad)));
}

// Radiance + visibility with cascade-0 priority (occupancy not needed for
// grass — blades are too thin for voxel AO to read).
void sampleCascades(vec3 wp, out vec3 radiance, out vec3 vis) {
    vec3 uv0 = c_uv(wp, u_c0_origin_m, u_c0_cell_m, u_c0_cells);
    if (inBox(uv0, 0.02)) {
        radiance = texture(u_c0_radiance, uv0).rgb;
        vis      = texture(u_c0_vis, uv0).rgb;
        return;
    }
    vec3 uv1 = c_uv(wp, u_c1_origin_m, u_c1_cell_m, u_c1_cells);
    if (inBox(uv1, 0.01)) {
        radiance = texture(u_c1_radiance, uv1).rgb;
        vis      = texture(u_c1_vis, uv1).rgb;
        return;
    }
    radiance = u_sky_ambient * 0.6;
    vis      = vec3(1.0);
}

vec3 acesTonemap(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14),
                 0.0, 1.0);
}

void main() {
    vec4 albedo = texture(u_tuft, v_uv);
    if (albedo.a < 0.5) discard;      // binary cutout — no blending/sorting

    vec3 base = pow(albedo.rgb, vec3(2.2)) * v_tint;

    // Light at the blade base, snapped to the same light-pixel grid as the
    // terrain so a tuft and the ground it stands on share light patches.
    vec3 wq = (floor(v_base_world / u_quant_m) + 0.5) * u_quant_m;
    vec3 radiance, vis;
    sampleCascades(wq + vec3(0.0, 0.0, 0.75), radiance, vis);

    // Blades are vertical: Lambert against straight-up normals.
    vec3 direct = u_sun_radiance  * (vis.r * max(u_sun_dir.z,  0.0))
                + u_moon_radiance * (vis.g * max(u_moon_dir.z, 0.0));

    vec3 hdr = base * (direct + radiance);

    if (u_fog_enabled > 0.5) {
        float dist = length(v_base_world - u_cam_pos);
        float w = log(max(dist, u_fog_near) / u_fog_near)
                / log(u_fog_far / u_fog_near);
        vec2 suv = gl_FragCoord.xy / u_viewport;
        vec4 fog = texture(u_fog_integrated, vec3(suv, clamp(w, 0.0, 1.0)));
        hdr = hdr * fog.a + fog.rgb;
    }

    vec3 ldr = acesTonemap(hdr * u_exposure);
    frag_color = vec4(pow(ldr, vec3(1.0 / 2.2)), 1.0);
}
"""
