#version 330 core
uniform mat4 p3d_ModelViewProjectionMatrix;

// --- per-volume (set by LeafLitterComponent, one node per "trees" volume) ---
uniform vec3  u_bounds_min;       // volume AABB min corner (world m)
uniform vec3  u_bounds_max;
uniform int   u_hash_seed;        // for_domain("wind","leaves",vol.id)
uniform float u_leaf_size_m;      // leaf billboard size (m)
uniform float u_leaf_life_s;      // looping lifetime (s)

// --- shared clock + camera + wind field (inherited from terrain_root) -------
uniform float u_time_s;
uniform vec3  u_cam_pos;
uniform sampler2D u_wind_tex;     // R=vx G=vy B=turb A=horizontal speed (m/s)
uniform vec2  u_wind_origin;
uniform float u_wind_cell_m;
uniform float u_wind_cells;
uniform float u_wind_enabled;

in vec4 p3d_Vertex;               // quad-local corner: xy in [-1,1], z=0
in vec2 p3d_MultiTexCoord0;

out vec2  v_uv;                   // atlas-remapped UV (one of 3 leaf cells)
out vec3  v_base_world;           // lighting sample point (leaf centre)

// lowbias32 — same hash as grass.vert / grass_placement.py.
uint lowbias32(uint x) {
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}
float u2f(uint h) { return float(h) * (1.0 / 4294967296.0); }

// Rotate vector v by angle a about a unit axis (Rodrigues).
vec3 rotAxis(vec3 v, vec3 axis, float a) {
    float c = cos(a), s = sin(a);
    return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0 - c);
}

void main() {
    uint i  = uint(gl_InstanceID);
    uint h0 = lowbias32(i  ^ uint(u_hash_seed));
    uint h1 = lowbias32(h0 ^ 0x9e3779b9u);
    uint h2 = lowbias32(h1 ^ 0x85ebca6bu);
    uint h3 = lowbias32(h2 ^ 0xc2b2ae35u);
    uint h4 = lowbias32(h3 ^ 0x27d4eb2fu);
    uint h5 = lowbias32(h4 ^ 0x165667b1u);

    vec3 size = u_bounds_max - u_bounds_min;

    // Home position inside the volume.  XY uniform; Z biased LOW (square the
    // hash so litter mostly sits near the ground and only some rides high).
    float zlow = u2f(h2); zlow = zlow * zlow;
    vec3 home = u_bounds_min + vec3(u2f(h0), u2f(h1), zlow) * size;

    // Looping life recycles each leaf back to its home; hashed phase staggers
    // them so they don't all reset on the same frame.
    float life = fract(u_time_s / u_leaf_life_s + u2f(h3));

    // Wind carry: sample the inherited field at the leaf home.  Carry strength
    // ramps with gust speed — calm air ⇒ litter settles (~0.3×), gusts/storms
    // ⇒ it streams (→1.0×).  Multiply by life so leaves stream out then recycle.
    vec3 carry = vec3(0.0);
    float gust = 0.0;
    if (u_wind_enabled > 0.5) {
        vec2 uv = (home.xy - u_wind_origin) / (u_wind_cell_m * u_wind_cells);
        vec4 w  = texture(u_wind_tex, uv);
        gust = clamp(w.a / 12.0, 0.0, 1.0);
        float k = 0.3 + 0.7 * gust;
        carry.xy += w.xy * k * (life * u_leaf_life_s);
        // A little lift in gusts so leaves kick UP off the ground, not just slide.
        carry.z  += (0.4 * gust + 0.3 * w.b) * sin(life * 3.14159265) * 2.0;
    } else {
        carry.xy += vec2(0.6, 0.0) * (life * u_leaf_life_s);   // faint flat drift
    }

    vec3 center = home + carry;

    // Tumble: two hashed angular rates about two hashed axes, composed — gives
    // a leaf a chaotic spin as it falls/streams.  Rates scale a touch with gust.
    float spin = 1.0 + 2.5 * gust;
    float a0 = u_time_s * (0.8 + 1.6 * u2f(h4)) * spin + u2f(h0) * 6.2831853;
    float a1 = u_time_s * (0.6 + 1.4 * u2f(h5)) * spin + u2f(h1) * 6.2831853;
    vec3 ax0 = normalize(vec3(u2f(h4) - 0.5, u2f(h5) - 0.5, u2f(h0) - 0.5) + 0.001);
    vec3 ax1 = normalize(vec3(u2f(h1) - 0.5, u2f(h2) - 0.5, u2f(h3) - 0.5) + 0.001);

    // Leaf quad corner in local space, tumbled by the two rotations.
    float sz = u_leaf_size_m * (0.7 + 0.6 * u2f(h5));
    vec3 corner = vec3(p3d_Vertex.xy * sz, 0.0);
    corner = rotAxis(corner, ax0, a0);
    corner = rotAxis(corner, ax1, a1);

    vec3 wp = center + corner;
    gl_Position = p3d_ModelViewProjectionMatrix * vec4(wp, 1.0);

    // Atlas UV: pick one of the 3 leaf variants from a hash → remap U into the
    // chosen 1/3 column.  variant ∈ {0,1,2}.
    float variant = floor(u2f(h0) * 3.0);
    variant = min(variant, 2.0);
    v_uv = vec2((variant + p3d_MultiTexCoord0.x) / 3.0, p3d_MultiTexCoord0.y);

    // Lighting sample point: the leaf centre (matches grass's base-world tap).
    v_base_world = center;
}
