#version 330 core
// flora.vert — GPU-instanced flowers / bushes / trees (world/flora_renderer.py).
//
// The grass instance idiom, generalised: the hash chain below is the grass
// chain PLUS one link (h5 → sprite-atlas variant), and the sway is shaped by
// two per-kind uniforms (u_sway_gain scales amplitude, u_sway_pivot sets the
// normalised height where the plant starts bending — 0.0 bends like a grass
// blade, ~0.45 pins a tree trunk and sways only the canopy).
uniform mat4 p3d_ModelViewProjectionMatrix;

// --- per-volume (set by FloraRendererComponent) ---------------------------
uniform vec3  u_bounds_min;       // volume AABB min corner (world m)
uniform vec3  u_bounds_max;
uniform int   u_hash_seed;        // per-(volume, kind) seed (zones/flora_placement.py)
uniform sampler2D u_height_field; // R: surface height in z-window; 255=none

// --- per-kind tuning (config [flora]) --------------------------------------
uniform float u_plant_height_m;   // unscaled sprite/quad height
uniform float u_fade_start_m;
uniform float u_fade_end_m;
uniform float u_scale_min;        // per-instance size jitter range
uniform float u_scale_span;
uniform float u_sway_gain;        // amplitude multiplier vs grass (1.0 = grass)
uniform float u_sway_pivot;       // normalised height where bending starts
uniform float u_variants;         // sprite atlas cell count

// --- weather sway (per frame from SkyState) — SCALAR FALLBACK --------------
// Same contract as grass.vert: used when the wind field is off.
uniform vec2  u_wind_dir;
uniform float u_sway_base;
uniform float u_sway_gust;
uniform float u_gust_freq;
uniform float u_time_s;

// --- wind field (inherited from terrain_root; WindSystemComponent) ---------
uniform sampler2D u_wind_tex;     // R=vx G=vy B=turb A=horizontal speed (m/s)
uniform vec2  u_wind_origin;
uniform float u_wind_cell_m;
uniform float u_wind_cells;
uniform float u_wind_enabled;

// --- shared lighting contract (inherited from terrain_root) ----------------
uniform vec3  u_cam_pos;

in vec4 p3d_Vertex;               // plant-local position (z up, base at 0)
in vec2 p3d_MultiTexCoord0;       // cell-local UV (x 0..1 across ONE cell)

out vec2  v_uv;                   // atlas UV (x remapped into the variant cell)
out vec3  v_base_world;           // plant base (lighting sample point)
out float v_tint;                 // per-instance albedo jitter

// lowbias32 (Chris Wellons) — LINE-FOR-LINE mirror of
// zones/flora_placement.py::flora_instance_attribs.  Edit both or neither.
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
    // Hash chain — mirror of zones/flora_placement.py::flora_instance_attribs.
    uint i  = uint(gl_InstanceID);
    uint h0 = lowbias32(i  ^ uint(u_hash_seed));
    uint h1 = lowbias32(h0 ^ 0x9e3779b9u);
    uint h2 = lowbias32(h1 ^ 0x85ebca6bu);
    uint h3 = lowbias32(h2 ^ 0xc2b2ae35u);
    uint h4 = lowbias32(h3 ^ 0x27d4eb2fu);
    uint h5 = lowbias32(h4 ^ 0x165667b1u);

    vec3 size = u_bounds_max - u_bounds_min;
    vec2 base_xy = u_bounds_min.xy + vec2(u2f(h0), u2f(h1)) * size.xy;

    // Terrain surface under this plant (baked field; 255 = no ground).
    vec2 field_uv = (base_xy - u_bounds_min.xy) / size.xy;
    float r = texture(u_height_field, field_uv).r;

    // Distance fade: shrink to zero between fade_start and fade_end.
    float fade = 1.0 - smoothstep(u_fade_start_m, u_fade_end_m,
                                  distance(base_xy, u_cam_pos.xy));

    if (r * 255.0 > 254.5 || fade <= 0.001) {
        gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
        v_uv = vec2(0.0);
        v_base_world = vec3(0.0);
        v_tint = 1.0;
        return;
    }

    float base_z = u_bounds_min.z + (r * 255.0 / 254.0) * size.z;

    // Per-instance yaw + scale jitter, shrunk by the distance fade.
    float rot = u2f(h2) * 6.2831853;
    float scale = (u_scale_min + u_scale_span * u2f(h3)) * fade;
    float c = cos(rot), s = sin(rot);
    vec2 lp = vec2(c * p3d_Vertex.x - s * p3d_Vertex.y,
                   s * p3d_Vertex.x + c * p3d_Vertex.y);

    // Sway weight: 0 below the pivot (trunk pinned), rising to 1 at the top.
    // Squared so the bend reads as a curve, not a hinge.
    float hn = clamp(p3d_Vertex.z / u_plant_height_m, 0.0, 1.0);
    float w_sway = smoothstep(u_sway_pivot, 1.0, hn);
    w_sway *= w_sway;
    float phase = u2f(h4) * 6.2831853;

    vec3 wp = vec3(base_xy + lp * scale,
                   base_z + p3d_Vertex.z * scale);

    if (u_wind_enabled > 0.5) {
        // Wind field: each plant samples its OWN local wind (same decode as
        // grass.vert) so gust bands travel across a flower meadow and roll
        // through a treeline canopy by canopy.
        vec2 uv = (base_xy - u_wind_origin) / (u_wind_cell_m * u_wind_cells);
        vec4 w  = texture(u_wind_tex, uv);            // R=vx G=vy B=turb A=speed
        float sn  = clamp(w.a / 12.0, 0.0, 1.0);
        vec2 dir  = (w.a > 1e-3) ? w.xy / w.a : u_wind_dir;
        float lean = (0.02 + 0.16 * sn) * (0.6 + 0.8 * u2f(h2))
                   + (0.03 + 0.18 * sn + 0.12 * w.b)
                   * sin(u_time_s * (1.2 + 0.25 * w.a) + phase);
        wp.xy += dir * (lean * u_sway_gain * w_sway);
    } else {
        // Scalar fallback (CPU backend / wind disabled) — grass.vert verbatim,
        // scaled by the kind's sway gain.
        float lean = u_sway_base * (0.6 + 0.8 * u2f(h2))
                   + u_sway_gust * sin(u_time_s * u_gust_freq + phase);
        wp.xy += u_wind_dir * (lean * u_sway_gain * w_sway);
    }

    gl_Position = p3d_ModelViewProjectionMatrix * vec4(wp, 1.0);

    // Atlas variant: remap the cell-local U into cell h5 % u_variants.
    float variant = float(h5 % uint(max(u_variants, 1.0)));
    v_uv = vec2((variant + p3d_MultiTexCoord0.x) / u_variants,
                p3d_MultiTexCoord0.y);
    v_base_world = vec3(base_xy, base_z);
    v_tint = 0.85 + 0.30 * u2f(h4);
}
