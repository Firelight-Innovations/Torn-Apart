#version 330 core
uniform mat4 p3d_ModelViewMatrix;
uniform mat4 p3d_ProjectionMatrix;

// --- per-component (set by DustMoteComponent) ---------------------------
uniform int   u_hash_seed;        // for_domain("wind","motes") draw
uniform float u_mote_box_m;       // camera-anchored lattice cell edge (m)
uniform float u_mote_size_m;      // billboard half-extent base (m)
uniform float u_mote_life_s;      // looping lifetime (s)
uniform vec3  u_cam_pos;          // camera world pos (shared lighting uniform,
                                  // refreshed each frame by DustMoteComponent)

// --- shared clock + wind field (inherited from terrain_root) ------------
uniform float u_time_s;
uniform sampler2D u_wind_tex;     // R=vx G=vy B=turb A=horizontal speed (m/s)
uniform vec2  u_wind_origin;
uniform float u_wind_cell_m;
uniform float u_wind_cells;
uniform float u_wind_enabled;     // 0.0 = flat-drift fallback, 1.0 = sample

in vec4 p3d_Vertex;               // quad-local corner: xy in [-1,1], z=0
in vec2 p3d_MultiTexCoord0;

out vec2  v_uv;
out float v_fade;                 // life × edge fade → fragment opacity

// lowbias32 (Chris Wellons) — same hash as grass.vert / grass_placement.py.
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
    // Per-instance hash chain from gl_InstanceID + component seed.
    uint i  = uint(gl_InstanceID);
    uint h0 = lowbias32(i  ^ uint(u_hash_seed));
    uint h1 = lowbias32(h0 ^ 0x9e3779b9u);
    uint h2 = lowbias32(h1 ^ 0x85ebca6bu);
    uint h3 = lowbias32(h2 ^ 0xc2b2ae35u);
    uint h4 = lowbias32(h3 ^ 0x27d4eb2fu);

    // Camera-anchored wrapping lattice: anchor the home cell to a box-grid
    // snapped under the camera, so as the camera moves motes tile space and
    // recycle with NO spawn pop (the anchor jumps a whole box at a time; the
    // hashed in-cell offset is camera-independent).
    float box = u_mote_box_m;
    vec3 anchor = floor(u_cam_pos / box) * box;
    vec3 home = anchor + vec3(u2f(h0), u2f(h1), u2f(h2)) * box;

    // Looping life in [0,1): each mote has a hashed phase so they don't all
    // pulse together.  sin(life*PI) fades in at birth and out at death.
    float life = fract(u_time_s / u_mote_life_s + u2f(h3));
    float life_fade = sin(life * 3.14159265);

    // Wind advection: sample the inherited field at the mote's home XY.
    vec3 disp = vec3(0.0);
    float turb = 0.0;
    if (u_wind_enabled > 0.5) {
        vec2 uv = (home.xy - u_wind_origin) / (u_wind_cell_m * u_wind_cells);
        vec4 w  = texture(u_wind_tex, uv);     // R=vx G=vy B=turb A=speed
        turb = w.b;
        // Local wind carries the mote downwind across its life; the multiply by
        // life means each mote streaks from its home cell and recycles.
        disp.xy += w.xy * (life * u_mote_life_s);
        // Gentle rise from turbulence (motes lift in gusty air).
        disp.z  += (0.25 + 0.6 * turb) * life;
    } else {
        // Flat-drift fallback (CPU backend / wind off): a slow +X breeze so
        // motes still drift instead of hanging dead-still.
        disp.xy += vec2(1.2, 0.0) * (life * u_mote_life_s);
    }

    // Hashed Brownian jitter, re-randomised on a coarse time step so the path
    // wiggles rather than tracking a straight line.  hash(i, floor(time*k)).
    uint tstep = uint(int(u_time_s * 1.5));
    uint hj0 = lowbias32(i ^ (tstep * 0x9e3779b9u));
    uint hj1 = lowbias32(hj0 ^ 0xc2b2ae35u);
    float jit = u_mote_box_m * 0.08;
    disp.xy += (vec2(u2f(hj0), u2f(hj1)) - 0.5) * jit;

    vec3 center = home + disp;

    // Box-edge fade: keep the lattice from showing hard popping at the wrap
    // boundary — motes dim as they near the cell-box edges around the anchor.
    vec3 rel = (center - anchor) / box;        // ~[0,1] (plus the disp spill)
    vec3 ef = min(rel, 1.0 - rel) * 4.0;       // 0 at edges, ramps to 1 inward
    float edge_fade = clamp(min(min(ef.x, ef.y), ef.z), 0.0, 1.0);
    v_fade = life_fade * edge_fade;

    // Billboard in view space: project the world centre into eye space, then
    // offset by the quad corner so the speck always faces the camera at a
    // constant screen-relative size.  p3d_Vertex.xy in [-1,1].
    float sz = u_mote_size_m * (0.7 + 0.6 * u2f(h4));
    vec4 view_center = p3d_ModelViewMatrix * vec4(center, 1.0);
    view_center.xy += p3d_Vertex.xy * sz;
    gl_Position = p3d_ProjectionMatrix * view_center;

    v_uv = p3d_MultiTexCoord0;
}
