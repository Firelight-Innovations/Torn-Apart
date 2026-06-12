#version 330 core
// tree.vert — instanced 3-D tree/bush meshes (world/tree_renderer.py).
//
// Unlike grass/flora there is NO hash chain here: placement is CPU-baked
// (zones/tree_placement.py) and packed into a small RGBA32F DATA TEXTURE,
// fetched per instance:
//     texel (0, i) = (x, y, z, yaw)
//     texel (1, i) = (scale, phase, tint, variant)
// instances_data_block() writes EXACTLY this layout — edit both or neither
// (tests/test_tree_placement.py pins it).
//
// Wind: the per-VERTEX sway weight baked into p3d_Color.a (0 trunk base →
// ≈1 leaf tips, procedural/flora/mesher.py) bends canopies while pinning
// trunks; the wind sampling itself is the flora.vert dual path (field
// texture when enabled, scalar SkyState fallback otherwise).
uniform mat4 p3d_ModelViewProjectionMatrix;

// --- per-draw (set by TreeRendererComponent) -------------------------------
uniform sampler2D u_inst_tex;     // RGBA32F, 2 texels/instance (see header)
uniform float u_fade_start_m;     // mesh shrink-away window (impostor fades
uniform float u_fade_end_m;       //  IN over the same window)
uniform float u_sway_gain;        // canopy sway amplitude per kind

// --- weather sway (per frame from SkyState) — SCALAR FALLBACK --------------
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

uniform vec3  u_cam_pos;

in vec4 p3d_Vertex;               // tree-local position (Z up, base at 0)
in vec3 p3d_Normal;               // flat face normal (tree-local)
in vec2 p3d_MultiTexCoord0;       // species-atlas UV (bark/leaf rect)
in vec4 p3d_Color;                // rgb = baked variant tint, A = SWAY WEIGHT

out vec2  v_uv;
out vec3  v_normal;               // world-space (yaw-rotated)
out vec3  v_world;                // fragment world position (lighting/fog)
out vec3  v_base_world;           // trunk base (fog distance anchor)
out float v_tint;

void main() {
    int i = gl_InstanceID;
    vec4 t0 = texelFetch(u_inst_tex, ivec2(0, i), 0);   // x, y, z, yaw
    vec4 t1 = texelFetch(u_inst_tex, ivec2(1, i), 0);   // scale, phase, tint, variant

    // Distance fade: shrink to zero — the grass cull idiom (degenerate
    // clip position for fully-faded instances).
    float fade = 1.0 - smoothstep(u_fade_start_m, u_fade_end_m,
                                  distance(t0.xy, u_cam_pos.xy));
    if (fade <= 0.001) {
        gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
        v_uv = vec2(0.0);
        v_normal = vec3(0.0, 0.0, 1.0);
        v_world = vec3(0.0);
        v_base_world = vec3(0.0);
        v_tint = 1.0;
        return;
    }

    // Yaw-rotate position AND normal, scale, translate onto the terrain.
    float c = cos(t0.w), s = sin(t0.w);
    float scale = t1.x * fade;
    vec3 lp = vec3(c * p3d_Vertex.x - s * p3d_Vertex.y,
                   s * p3d_Vertex.x + c * p3d_Vertex.y,
                   p3d_Vertex.z);
    vec3 wp = t0.xyz + lp * scale;

    // Wind lean, weighted by the baked per-vertex sway (squared → curve,
    // not hinge).  rnd de-synchronises neighbours' static lean.
    float w = p3d_Color.a * p3d_Color.a;
    float rnd = fract(t1.y * 0.15915494);        // phase / 2π
    if (u_wind_enabled > 0.5) {
        // Each tree samples its OWN local wind (flora.vert decode) so gust
        // bands roll through a treeline canopy by canopy.
        vec2 uv = (t0.xy - u_wind_origin) / (u_wind_cell_m * u_wind_cells);
        vec4 wf = texture(u_wind_tex, uv);       // R=vx G=vy B=turb A=speed
        float sn  = clamp(wf.a / 12.0, 0.0, 1.0);
        vec2 dir  = (wf.a > 1e-3) ? wf.xy / wf.a : u_wind_dir;
        float lean = (0.02 + 0.16 * sn) * (0.6 + 0.8 * rnd)
                   + (0.03 + 0.18 * sn + 0.12 * wf.b)
                   * sin(u_time_s * (1.2 + 0.25 * wf.a) + t1.y);
        wp.xy += dir * (lean * u_sway_gain * w);
    } else {
        float lean = u_sway_base * (0.6 + 0.8 * rnd)
                   + u_sway_gust * sin(u_time_s * u_gust_freq + t1.y);
        wp.xy += u_wind_dir * (lean * u_sway_gain * w);
    }

    gl_Position = p3d_ModelViewProjectionMatrix * vec4(wp, 1.0);
    v_uv = p3d_MultiTexCoord0;
    v_normal = vec3(c * p3d_Normal.x - s * p3d_Normal.y,
                    s * p3d_Normal.x + c * p3d_Normal.y,
                    p3d_Normal.z);
    v_world = wp;
    v_base_world = t0.xyz;
    v_tint = t1.z * p3d_Color.r;     // per-instance tint × baked variant tint
}
