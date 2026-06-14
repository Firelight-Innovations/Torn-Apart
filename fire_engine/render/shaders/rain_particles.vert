#version 330 core
// rain_particles.vert — GPU-instanced volumetric rain streaks (M6).
//
// Sibling of mote_dust.vert: a camera-anchored WRAPPING LATTICE of streaks
// (instance XY/Z from a gl_InstanceID hash, wrapped into a box around the
// camera) FALLING via u_time_s and swaying with the inherited wind texture.
// Two per-instance gates decide whether a streak draws at all:
//
//   * HEIGHTMAP CULL (the M6 fix): sample the rain-cover heightmap at the
//     instance world XY; if its world Z is BELOW the cover height there (under
//     a roof/overhang), collapse the quad to zero size so it never rasterises.
//   * STORM-FOOTPRINT GATE: sample the weather-map precip channel (B) at the
//     instance world XY; rain only exists where precip > 0 (fade with precip),
//     so rain is bounded to storm cells, not global.  When the weather map is
//     off, fall back to the scalar u_rain_intensity (SkyState.rain_intensity).

uniform mat4 p3d_ModelViewMatrix;
uniform mat4 p3d_ProjectionMatrix;

// --- per-component (set by RainRendererComponent) -----------------------
uniform int   u_hash_seed;        // for_domain("rain","particles") draw
uniform float u_rain_box_m;       // camera-anchored lattice cell edge (m)
uniform float u_rain_size_m;      // streak half-width base (m)
uniform float u_rain_length_m;    // streak length base (m)
uniform float u_rain_fall_mps;    // base fall speed (m/s)
uniform float u_rain_intensity;   // scalar fallback when the weather map is off
uniform float u_rain_occlusion;   // 1.0 = apply the heightmap cull, 0.0 = off
uniform vec3  u_cam_pos;          // camera world pos (refreshed each frame)

// --- rain-cover heightmap (set by RainRendererComponent) ----------------
uniform sampler2D u_rain_height_tex;   // R = world Z (m) of the highest solid
uniform vec2  u_rain_height_origin;    // min-corner world XY (m) of texel (0,0)
uniform float u_rain_height_cell_m;    // column edge (m)
uniform float u_rain_height_cells;     // columns per axis

// --- shared clock + wind field (inherited from terrain_root) ------------
uniform float u_time_s;
uniform sampler2D u_wind_tex;     // R=vx G=vy B=turb A=horizontal speed (m/s)
uniform vec2  u_wind_origin;
uniform float u_wind_cell_m;
uniform float u_wind_cells;
uniform float u_wind_enabled;

// --- weather map (inherited from render: R=cov G=den B=precip A=fog) -----
uniform sampler2D u_weather_map;
uniform vec2  u_wmap_origin;
uniform float u_wmap_cell_m;
uniform float u_wmap_cells;
uniform int   u_weather_map_enabled;

in vec4 p3d_Vertex;               // quad-local corner: xy in [-1,1], z=0
in vec2 p3d_MultiTexCoord0;

out vec2  v_uv;
out float v_fade;                 // precip × edge fade → fragment opacity

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
    uint i  = uint(gl_InstanceID);
    uint h0 = lowbias32(i  ^ uint(u_hash_seed));
    uint h1 = lowbias32(h0 ^ 0x9e3779b9u);
    uint h2 = lowbias32(h1 ^ 0x85ebca6bu);
    uint h3 = lowbias32(h2 ^ 0xc2b2ae35u);
    uint h4 = lowbias32(h3 ^ 0x27d4eb2fu);

    // Camera-anchored wrapping lattice (same trick as the dust motes): the home
    // cell snaps to a box grid under the camera, so streaks tile space and
    // recycle with no spawn pop as the camera flies.  Bias the box UP so most
    // streaks live above the camera (rain falls from overhead).
    float box = u_rain_box_m;
    vec3 anchor = floor(u_cam_pos / box) * box;
    vec3 home = anchor + vec3(u2f(h0), u2f(h1), u2f(h2)) * box;

    // Fall: each streak loops down by u_rain_fall_mps over the box height, with
    // a hashed phase so they don't all reset together.  Subtract from Z; wrap
    // within the box with fract so the column recycles seamlessly.
    float speed = u_rain_fall_mps * (0.8 + 0.4 * u2f(h3));
    float fall = fract(u_time_s * speed / box + u2f(h4)) * box;
    vec3 center = home;
    center.z = anchor.z + box - fall;     // start high, fall toward the anchor

    // Wind sway: nudge the streak downwind so rain slants in a gale.  Sampled
    // at the streak XY from the inherited wind field (clamped outside region).
    if (u_wind_enabled > 0.5) {
        vec2 uv = (center.xy - u_wind_origin) / (u_wind_cell_m * u_wind_cells);
        vec4 w  = texture(u_wind_tex, uv);
        center.xy += w.xy * 0.18;          // slant: a fraction of the wind speed
    }

    // --- Storm-footprint gate: precip at the streak XY -------------------
    float precip;
    if (u_weather_map_enabled != 0) {
        vec2 wuv = (center.xy - u_wmap_origin) / (u_wmap_cell_m * u_wmap_cells);
        precip = texture(u_weather_map, wuv).b;      // B = precip channel
    } else {
        precip = u_rain_intensity;                   // flat scalar fallback
    }

    // --- Heightmap cull (THE FIX): kill streaks under a roof/overhang ----
    // Sample the cover height at the streak XY; if the streak is below it, it
    // is inside/under solid cover → collapse the quad so it never draws.
    float under_cover = 0.0;
    if (u_rain_occlusion > 0.5) {
        vec2 huv = (center.xy - u_rain_height_origin)
                   / (u_rain_height_cell_m * u_rain_height_cells);
        // Only cull when the sample is inside the window; outside, assume open.
        if (huv.x >= 0.0 && huv.x <= 1.0 && huv.y >= 0.0 && huv.y <= 1.0) {
            float cover_z = texture(u_rain_height_tex, huv).r;
            if (center.z < cover_z) under_cover = 1.0;
        }
    }

    // Cull conditions → zero size (degenerate quad, never rasterises).
    float alive = step(0.02, precip) * (1.0 - under_cover);

    // Box-edge fade so streaks dim near the wrap boundary (no hard pop).
    vec3 rel = (center - anchor) / box;
    vec3 ef = min(rel, 1.0 - rel) * 4.0;
    float edge_fade = clamp(min(min(ef.x, ef.y), ef.z), 0.0, 1.0);
    v_fade = clamp(precip, 0.0, 1.0) * edge_fade * alive;

    // Billboard a vertical streak in VIEW space: width = u_rain_size_m, height
    // = u_rain_length_m (scaled a touch by intensity so heavy rain streaks
    // longer).  p3d_Vertex.xy in [-1,1]: x = half-width, y = half-length.
    float w_half = u_rain_size_m * alive;
    float l_half = u_rain_length_m * (0.7 + 0.6 * clamp(precip, 0.0, 1.0)) * alive;
    vec4 view_center = p3d_ModelViewMatrix * vec4(center, 1.0);
    view_center.x += p3d_Vertex.x * w_half;
    view_center.y += p3d_Vertex.y * l_half;
    gl_Position = p3d_ProjectionMatrix * view_center;

    v_uv = p3d_MultiTexCoord0;
}
