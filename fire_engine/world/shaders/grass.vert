#version 330 core
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

// lowbias32 (Chris Wellons) â€” LINE-FOR-LINE mirror of
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
    // Hash chain â€” mirror of zones/grass_placement.py::instance_attribs.
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
        // outside the frustum â€” zero-area triangles, no fragments.
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
