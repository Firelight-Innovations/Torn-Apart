
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
uniform float u_hdr_output;   // 1.0 → linearize for the HDR buffer; 0.0 → legacy LDR

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
    // second box is an interior face that should not be visible â€” without
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
    vec3 cloud_col = col / max(acc, 1e-4);           // un-premultiply for M_alpha
    // These boxy clouds use display-referred flat face colours.  When the scene
    // renders into the linear-HDR buffer, linearise so the alpha blend over the
    // (linear) sky reads correctly.  (Interim: the volumetric cloud rewrite
    // replaces this shader entirely.)
    if (u_hdr_output > 0.5) cloud_col = pow(cloud_col, vec3(2.2));
    frag_color = vec4(cloud_col, acc);
}
