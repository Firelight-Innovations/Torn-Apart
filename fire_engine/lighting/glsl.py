"""
lighting/glsl.py — GLSL compute-shader sources for the GPU lighting pipeline.

Plain string constants (this module imports nothing GPU-side, so it stays
importable headless; only `lighting/gpu.py` compiles them via panda3d).

Pipeline shape (per radiance cascade)
-------------------------------------
1. **INJECT** (runs when the volume, sun, sky or lights change) — per cell:
   - march toward the sun / moon through occupancy → celestial visibility
     (``u_vis``: R = sun, G = moon), the source of all voxel shadows;
   - march straight up → sky visibility, injecting ``sky_ambient`` skylight
     into open-air cells;
   - first-bounce sunlight: an air cell next to a sunlit solid neighbour
     receives the neighbour's albedo × sun radiance × Lambert (this is what
     makes a sunlit red wall glow red into the room);
   - emissive solid neighbours leak their emission into adjacent air;
   - dynamic point/area lights add windowed-inverse-square radiance with a
     short occupancy march for shadows.
   Writes ``u_direct`` (rgba16f source radiance).

2. **PROPAGATE** (every frame, ``light_prop_iters`` ping-pong iterations) —
   exponential diffusion: ``next = direct·(1−decay) + decay·avg₆``, where a
   solid neighbour reflects the cell's own radiance back tinted by its
   albedo (multi-bounce approximation).  Spectral radius < 1 ⇒ stable; light
   visibly "flows" a few cells per frame after a change, converging to a
   smooth flood-fill GI field.

Froxel fog
----------
3. **FOG_SCATTER** — per froxel (screen-aligned X/Y, exponential depth Z):
   weather density × (sun/moon Henyey-Greenstein in-scatter shadowed through
   the cascade occupancy → god rays, plus isotropic sky/GI ambient).
4. **FOG_INTEGRATE** — front-to-back accumulation along each screen ray,
   writing per-slice (accumulated in-scatter RGB, transmittance A) so any
   surface can composite fog at its own depth with one texture tap.

All units meters; world coordinates Z-up.  ``u_origin_m``/``u_cell_m``/
``u_cells`` define each cascade's window (see `lighting/volume.py`).
"""

from __future__ import annotations

__all__ = [
    "INJECT_COMPUTE",
    "PROPAGATE_COMPUTE",
    "FOG_SCATTER_COMPUTE",
    "FOG_INTEGRATE_COMPUTE",
    "MAX_LIGHTS",
]

# Must match Config.light_max_point_lights' upper bound and LightSet.pack.
MAX_LIGHTS = 64


# ---------------------------------------------------------------------------
# 1. Injection: direct radiance + celestial visibility
# ---------------------------------------------------------------------------

INJECT_COMPUTE = """#version 430
layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout (rgba8)   uniform readonly  image3D u_geom;    // rgb albedo, a occupancy
layout (rgba8)   uniform readonly  image3D u_emis;    // rgb emission / EMISSION_SCALE
layout (rgba8)   uniform writeonly image3D u_vis;     // r sun vis, g moon vis, b sky vis
layout (rgba16f) uniform writeonly image3D u_direct;  // injected source radiance

uniform int   u_cells;
uniform vec3  u_sun_dir;        // unit, toward sun (world)
uniform vec3  u_sun_radiance;   // linear HDR RGB at ground
uniform vec3  u_moon_dir;
uniform vec3  u_moon_radiance;
uniform vec3  u_sky_ambient;    // hemispheric skylight irradiance
uniform float u_bounce;         // first-bounce gain [0,1]
uniform float u_emission_scale; // EMISSION_SCALE from volume.py
uniform int   u_num_lights;
uniform vec4  u_light_pos_r[64];  // xyz world pos / box centre, w falloff radius
uniform vec4  u_light_col_t[64];  // rgb color*intensity, w type (0 point, 1 area, 2 spot)
uniform vec4  u_light_ext[64];    // xyz box half extents (area) / beam dir (spot),
                                  // w cos(half cone angle) (spot)
uniform int   u_num_boxes;        // dynamic occluder AABBs (dev cubes, props)
uniform vec4  u_box_min[16];      // xyz world min corner (meters)
uniform vec4  u_box_max[16];      // xyz world max corner
uniform vec3  u_origin_m;       // world meters of texel (0,0,0) min corner
uniform float u_cell_m;

float occAt(ivec3 c) {
    if (any(lessThan(c, ivec3(0))) || any(greaterThanEqual(c, ivec3(u_cells))))
        return 0.0;                       // outside the window = open air
    return imageLoad(u_geom, c).a;
}

// Analytic ray-vs-AABB shadow test against the dynamic occluder boxes
// (objects not voxelised into u_geom: dev cubes, props).  World space.
// Returns 0.0 when any box blocks the segment [0.05, tmax] along rd.
float boxVis(vec3 ro, vec3 rd, float tmax) {
    for (int i = 0; i < u_num_boxes; i++) {
        vec3 inv = 1.0 / (rd + vec3(1e-7));
        vec3 ta = (u_box_min[i].xyz - ro) * inv;
        vec3 tb = (u_box_max[i].xyz - ro) * inv;
        vec3 tlo = min(ta, tb);
        vec3 thi = max(ta, tb);
        float tn = max(max(tlo.x, tlo.y), tlo.z);
        float tf = min(min(thi.x, thi.y), thi.z);
        if (tf >= max(tn, 0.05) && tn < tmax) return 0.0;
    }
    return 1.0;
}

// Fixed-step occupancy march from a cell centre; 1.0 = unobstructed.
// Chunky single-cell steps are deliberate — voxel-crisp shadows.
float marchVis(ivec3 cell, vec3 dir, int maxSteps) {
    vec3 p = vec3(cell) + 0.5 + dir * 1.2;   // hop out of the own cell
    for (int i = 0; i < maxSteps; i++) {
        ivec3 c = ivec3(floor(p));
        if (any(lessThan(c, ivec3(0))) ||
            any(greaterThanEqual(c, ivec3(u_cells))))
            return 1.0;                       // left the window: assume open
        if (imageLoad(u_geom, c).a > 0.5) return 0.0;
        p += dir;
    }
    return 1.0;
}

const ivec3 NEIGHBORS[6] = ivec3[6](
    ivec3( 1, 0, 0), ivec3(-1, 0, 0),
    ivec3( 0, 1, 0), ivec3( 0,-1, 0),
    ivec3( 0, 0, 1), ivec3( 0, 0,-1));

void main() {
    ivec3 c = ivec3(gl_GlobalInvocationID.xyz);
    if (any(greaterThanEqual(c, ivec3(u_cells)))) return;

    vec4 g = imageLoad(u_geom, c);
    bool solid = g.a > 0.5;
    vec3 wp = u_origin_m + (vec3(c) + 0.5) * u_cell_m;

    // --- celestial + sky visibility ------------------------------------
    // Voxel occupancy march × analytic dynamic-occluder boxes: anything in
    // either field cuts sun/moon/sky light (and therefore god rays + AO).
    float sunVis  = (u_sun_dir.z  > -0.05)
                    ? marchVis(c, u_sun_dir, 36) * boxVis(wp, u_sun_dir, 1e3)
                    : 0.0;
    float moonVis = (u_moon_dir.z > -0.05 &&
                     dot(u_moon_radiance, vec3(1.0)) > 1e-4)
                    ? marchVis(c, u_moon_dir, 36) * boxVis(wp, u_moon_dir, 1e3)
                    : 0.0;
    float skyVis  = marchVis(c, vec3(0.0, 0.0, 1.0), 24)
                    * boxVis(wp, vec3(0.0, 0.0, 1.0), 1e3);
    imageStore(u_vis, c, vec4(sunVis, moonVis, skyVis, 1.0));

    // --- direct radiance injection --------------------------------------
    vec3 direct = vec3(0.0);
    if (solid) {
        // Emissive solids carry their own radiance (visible surface glow);
        // their light enters the flood fill through adjacent air below.
        direct = imageLoad(u_emis, c).rgb * u_emission_scale;
        imageStore(u_direct, c, vec4(direct, 1.0));
        return;
    }

    // Skylight: open-air cells receive the hemispheric sky ambient.
    direct += u_sky_ambient * skyVis;

    // First bounce + emissive leak from the 6 solid neighbours.
    for (int i = 0; i < 6; i++) {
        ivec3 nc = c + NEIGHBORS[i];
        float no = occAt(nc);
        if (no < 0.5) continue;
        vec3 nrm = -vec3(NEIGHBORS[i]);   // neighbour surface normal ≈ toward us
        vec4 ng = imageLoad(u_geom, clamp(nc, ivec3(0), ivec3(u_cells - 1)));
        float lamS = max(dot(nrm, u_sun_dir), 0.0);
        float lamM = max(dot(nrm, u_moon_dir), 0.0);
        direct += ng.rgb * (u_sun_radiance  * (sunVis  * lamS) +
                            u_moon_radiance * (moonVis * lamM)) *
                  (u_bounce * 0.45);
        direct += imageLoad(u_emis,
                  clamp(nc, ivec3(0), ivec3(u_cells - 1))).rgb *
                  (u_emission_scale * 0.6);
    }

    // Dynamic point / area / spot lights (windowed inverse-square +
    // shadow march + dynamic-occluder boxes).
    for (int i = 0; i < u_num_lights; i++) {
        vec3  lp = u_light_pos_r[i].xyz;
        float radius = u_light_pos_r[i].w;
        float ltype = u_light_col_t[i].w;
        if (ltype > 0.5 && ltype < 1.5) {
            // Area light: light from the closest point on the emissive box,
            // so distance/falloff is measured to the box surface.
            lp = clamp(wp, u_light_pos_r[i].xyz - u_light_ext[i].xyz,
                           u_light_pos_r[i].xyz + u_light_ext[i].xyz);
        }
        vec3  dv = lp - wp;
        float d  = length(dv);
        if (d >= radius) continue;
        float w  = d / radius;
        float win = 1.0 - w * w * w * w;            // (1 - (d/r)^4)
        float atten = (win * win) / (d * d + 1.0);
        if (ltype > 1.5) {
            // Spot cone: u_light_ext[i].xyz = beam dir, w = cos(half angle).
            // Smooth edge from the cone boundary toward its core.
            float cd = dot(normalize(-dv), u_light_ext[i].xyz);
            float cosO = u_light_ext[i].w;
            float spot = smoothstep(cosO, mix(cosO, 1.0, 0.35), cd);
            if (spot <= 0.001) continue;
            atten *= spot;
        }
        float vis = 1.0;
        if (d > u_cell_m * 1.5) {
            int steps = int(min(d / u_cell_m, 24.0));
            vec3 ld = dv / max(d, 1e-4);
            vis = marchVis(c, ld, steps) * boxVis(wp, ld, d - 0.1);
        }
        direct += u_light_col_t[i].rgb * (atten * vis);
    }

    imageStore(u_direct, c, vec4(direct, 1.0));
}
"""


# ---------------------------------------------------------------------------
# 2. Propagation: exponential diffusion flood fill (one Jacobi iteration)
# ---------------------------------------------------------------------------

PROPAGATE_COMPUTE = """#version 430
layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout (rgba16f) uniform readonly  image3D u_prev;    // radiance, last iter
layout (rgba16f) uniform readonly  image3D u_direct;  // injected sources
layout (rgba8)   uniform readonly  image3D u_geom;    // albedo + occupancy
layout (rgba16f) uniform writeonly image3D u_next;

uniform int   u_cells;
uniform float u_decay;    // diffusion weight [0,1): higher = light travels farther
uniform float u_bounce;   // wall-reflection gain [0,1]

const ivec3 NEIGHBORS[6] = ivec3[6](
    ivec3( 1, 0, 0), ivec3(-1, 0, 0),
    ivec3( 0, 1, 0), ivec3( 0,-1, 0),
    ivec3( 0, 0, 1), ivec3( 0, 0,-1));

void main() {
    ivec3 c = ivec3(gl_GlobalInvocationID.xyz);
    if (any(greaterThanEqual(c, ivec3(u_cells)))) return;

    vec3 direct = imageLoad(u_direct, c).rgb;
    vec4 g = imageLoad(u_geom, c);
    if (g.a > 0.5) {
        // Solid cells hold their own emission only; light flows around them.
        imageStore(u_next, c, vec4(direct, 1.0));
        return;
    }

    vec3 own = imageLoad(u_prev, c).rgb;
    vec3 sum = vec3(0.0);
    for (int i = 0; i < 6; i++) {
        ivec3 nc = c + NEIGHBORS[i];
        if (any(lessThan(nc, ivec3(0))) ||
            any(greaterThanEqual(nc, ivec3(u_cells)))) {
            sum += own;                      // open boundary: no gradient
            continue;
        }
        vec4 ng = imageLoad(u_geom, nc);
        if (ng.a > 0.5)
            sum += own * ng.rgb * u_bounce;  // wall bounces our light back, tinted
        else
            sum += imageLoad(u_prev, nc).rgb;
    }

    // next = direct·(1−decay) + decay·avg — converges to a diffusion of the
    // injected sources with reach ≈ 1/(1−decay) cells; always bounded.
    vec3 next = direct * (1.0 - u_decay) + (u_decay / 6.0) * sum;
    imageStore(u_next, c, vec4(next, 1.0));
}
"""


# ---------------------------------------------------------------------------
# 3. Froxel fog scatter
# ---------------------------------------------------------------------------

FOG_SCATTER_COMPUTE = """#version 430
layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout (rgba16f) uniform writeonly image3D u_froxels; // rgb inscatter·σ, a σ (extinction 1/m)

uniform ivec3 u_froxel_dim;       // (W, H, Z slices)
uniform vec3  u_cam_pos;          // world meters
uniform vec3  u_cam_fwd;          // unit basis (world)
uniform vec3  u_cam_right;
uniform vec3  u_cam_up;
uniform vec2  u_tan_half_fov;     // tan(fovx/2), tan(fovy/2)
uniform float u_fog_near;         // first-slice distance (m)
uniform float u_fog_far;          // last-slice distance (m)
uniform float u_fog_density;      // base extinction at ground level (1/m)
uniform float u_ground_z;         // height-falloff reference (m)
uniform float u_anisotropy;       // Henyey-Greenstein g
uniform vec3  u_sun_dir;
uniform vec3  u_sun_radiance;
uniform vec3  u_moon_dir;
uniform vec3  u_moon_radiance;
uniform vec3  u_sky_ambient;
// Cascade-1 lookups (largest range) for shadowing + GI ambient in the fog.
uniform sampler3D u_c1_vis;       // r sun, g moon, b sky visibility
uniform sampler3D u_c1_radiance;
uniform vec3  u_c1_origin_m;
uniform float u_c1_cell_m;
uniform float u_c1_cells;
uniform int   u_num_boxes;        // dynamic occluder AABBs (cut god rays)
uniform vec4  u_box_min[16];
uniform vec4  u_box_max[16];

// Same analytic ray-vs-AABB test as INJECT: dynamic objects shadow the fog.
float boxVis(vec3 ro, vec3 rd, float tmax) {
    for (int i = 0; i < u_num_boxes; i++) {
        vec3 inv = 1.0 / (rd + vec3(1e-7));
        vec3 ta = (u_box_min[i].xyz - ro) * inv;
        vec3 tb = (u_box_max[i].xyz - ro) * inv;
        vec3 tlo = min(ta, tb);
        vec3 thi = max(ta, tb);
        float tn = max(max(tlo.x, tlo.y), tlo.z);
        float tf = min(min(thi.x, thi.y), thi.z);
        if (tf >= max(tn, 0.05) && tn < tmax) return 0.0;
    }
    return 1.0;
}

float sliceDist(float s) {
    // Exponential slice distribution: equal screen-space fog detail near.
    return u_fog_near * pow(u_fog_far / u_fog_near,
                            s / float(u_froxel_dim.z));
}

float phaseHG(float cosT, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * 3.14159265 *
           pow(1.0 + g2 - 2.0 * g * cosT, 1.5));
}

vec3 c1uv(vec3 wp) {
    return (wp - u_c1_origin_m) / (u_c1_cell_m * u_c1_cells);
}

bool inC1(vec3 uv) {
    return all(greaterThan(uv, vec3(0.01))) && all(lessThan(uv, vec3(0.99)));
}

void main() {
    ivec3 f = ivec3(gl_GlobalInvocationID.xyz);
    if (any(greaterThanEqual(f, u_froxel_dim))) return;

    // Reconstruct the world-space sample point at this froxel's centre.
    vec2 ndc = (vec2(f.xy) + 0.5) / vec2(u_froxel_dim.xy) * 2.0 - 1.0;
    vec3 ray = normalize(u_cam_fwd +
                         u_cam_right * (ndc.x * u_tan_half_fov.x) +
                         u_cam_up    * (ndc.y * u_tan_half_fov.y));
    float d0 = sliceDist(float(f.z));
    float d1 = sliceDist(float(f.z) + 1.0);
    vec3 wp = u_cam_pos + ray * (0.5 * (d0 + d1));

    // Height-falloff exponential fog density.
    float h = max(wp.z - u_ground_z, 0.0);
    float sigma = u_fog_density * exp(-h / 28.0);

    // Visibility from the cascade (god rays: sun shafts cut by occupancy).
    vec3 uv = c1uv(wp);
    float sunVis = 1.0, moonVis = 1.0, skyVis = 1.0;
    vec3 gi = vec3(0.0);
    if (inC1(uv)) {
        vec3 vis = texture(u_c1_vis, uv).rgb;
        sunVis = vis.r; moonVis = vis.g; skyVis = vis.b;
        gi = texture(u_c1_radiance, uv).rgb;
    }
    // Dynamic occluders cut the celestial shafts (boxy god rays).
    if (u_num_boxes > 0) {
        sunVis  *= boxVis(wp, u_sun_dir, 1e3);
        moonVis *= boxVis(wp, u_moon_dir, 1e3);
    }

    float cosS = dot(ray, u_sun_dir);
    float cosM = dot(ray, u_moon_dir);
    vec3 inscatter =
        u_sun_radiance  * (sunVis  * phaseHG(cosS, u_anisotropy)) +
        u_moon_radiance * (moonVis * phaseHG(cosM, u_anisotropy * 0.8)) +
        (u_sky_ambient * skyVis + gi) * (0.25 / 3.14159265);

    imageStore(u_froxels, f, vec4(inscatter * sigma, sigma));
}
"""


# ---------------------------------------------------------------------------
# 4. Froxel fog integration (front-to-back along each screen ray)
# ---------------------------------------------------------------------------

FOG_INTEGRATE_COMPUTE = """#version 430
layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout (rgba16f) uniform readonly  image3D u_froxels;    // from FOG_SCATTER
layout (rgba16f) uniform writeonly image3D u_integrated; // rgb light, a transmittance

uniform ivec3 u_froxel_dim;
uniform float u_fog_near;
uniform float u_fog_far;

float sliceDist(float s) {
    return u_fog_near * pow(u_fog_far / u_fog_near,
                            s / float(u_froxel_dim.z));
}

void main() {
    ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(xy, u_froxel_dim.xy))) return;

    vec3 accum = vec3(0.0);
    float trans = 1.0;
    for (int z = 0; z < u_froxel_dim.z; z++) {
        vec4 s = imageLoad(u_froxels, ivec3(xy, z));
        float dz = sliceDist(float(z) + 1.0) - sliceDist(float(z));
        float stepTrans = exp(-s.a * dz);
        // Energy-conserving slice integration (Frostbite):
        vec3 sliceLight = (s.a > 1e-6)
            ? s.rgb * (1.0 - stepTrans) / s.a
            : vec3(0.0);
        accum += trans * sliceLight;
        trans *= stepTrans;
        imageStore(u_integrated, ivec3(xy, z), vec4(accum, trans));
    }
}
"""
