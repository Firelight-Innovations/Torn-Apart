#version 330 core
// tree_impostor.vert — far-LOD billboard stage of the 3-D tree pipeline.
//
// Reads the SAME per-instance data texture as tree.vert (layout pinned in
// zones/tree_placement.py) but with the OPPOSITE fade: the billboard grows
// in over the mesh window [u_mesh_fade_start_m, u_mesh_fade_end_m] and
// shrinks away over [u_fade_start_m, u_fade_end_m] — billboards are LOD
// ONLY.  The quad's atlas cell comes from the instance's variant (t1.w);
// the cells share one meters-per-texel so the quad overlays the mesh
// exactly at the crossfade.  Fragment stage: flora.frag verbatim.
uniform mat4 p3d_ModelViewProjectionMatrix;

// --- per-draw (set by TreeRendererComponent) -------------------------------
uniform sampler2D u_inst_tex;     // texel(0,i)=(x,y,z,yaw) texel(1,i)=(scale,phase,tint,variant)
uniform float u_mesh_fade_start_m;  // fade-IN window (mesh fades out here)
uniform float u_mesh_fade_end_m;
uniform float u_fade_start_m;       // fade-OUT window (gone past end)
uniform float u_fade_end_m;
uniform float u_variants;           // impostor atlas cell count
uniform float u_sway_gain;
uniform float u_sway_pivot;         // canopy-only sway (trunk pinned)

// --- weather sway (per frame from SkyState) — SCALAR FALLBACK --------------
uniform vec2  u_wind_dir;
uniform float u_sway_base;
uniform float u_sway_gust;
uniform float u_gust_freq;
uniform float u_time_s;

// --- wind field (inherited from terrain_root; WindSystemComponent) ---------
uniform sampler2D u_wind_tex;
uniform vec2  u_wind_origin;
uniform float u_wind_cell_m;
uniform float u_wind_cells;
uniform float u_wind_enabled;

uniform vec3  u_cam_pos;

in vec4 p3d_Vertex;               // crossed-quad local (Z up, base at 0)
in vec2 p3d_MultiTexCoord0;       // cell-local UV

out vec2  v_uv;                   // atlas UV (x remapped into variant cell)
out vec3  v_base_world;           // lighting sample anchor (flora.frag)
out float v_tint;

void main() {
    int i = gl_InstanceID;
    vec4 t0 = texelFetch(u_inst_tex, ivec2(0, i), 0);
    vec4 t1 = texelFetch(u_inst_tex, ivec2(1, i), 0);

    float d = distance(t0.xy, u_cam_pos.xy);
    float fade = smoothstep(u_mesh_fade_start_m, u_mesh_fade_end_m, d)
               * (1.0 - smoothstep(u_fade_start_m, u_fade_end_m, d));
    if (fade <= 0.001) {
        gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
        v_uv = vec2(0.0);
        v_base_world = vec3(0.0);
        v_tint = 1.0;
        return;
    }

    float c = cos(t0.w), s = sin(t0.w);
    float scale = t1.x * fade;
    vec2 lp = vec2(c * p3d_Vertex.x - s * p3d_Vertex.y,
                   s * p3d_Vertex.x + c * p3d_Vertex.y);
    vec3 wp = vec3(t0.xy + lp * scale, t0.z + p3d_Vertex.z * scale);

    // Canopy-only sway, exactly the retired tree-sprite behaviour: weight
    // by quad height above the pivot, squared for a curved bend.
    float hn = p3d_MultiTexCoord0.y;
    float w = smoothstep(u_sway_pivot, 1.0, hn);
    w *= w;
    if (u_wind_enabled > 0.5) {
        vec2 uv = (t0.xy - u_wind_origin) / (u_wind_cell_m * u_wind_cells);
        vec4 wf = texture(u_wind_tex, uv);
        float sn  = clamp(wf.a / 12.0, 0.0, 1.0);
        vec2 dir  = (wf.a > 1e-3) ? wf.xy / wf.a : u_wind_dir;
        float rnd = fract(t1.y * 0.15915494);
        float lean = (0.02 + 0.16 * sn) * (0.6 + 0.8 * rnd)
                   + (0.03 + 0.18 * sn + 0.12 * wf.b)
                   * sin(u_time_s * (1.2 + 0.25 * wf.a) + t1.y);
        wp.xy += dir * (lean * u_sway_gain * w);
    } else {
        float rnd = fract(t1.y * 0.15915494);
        float lean = u_sway_base * (0.6 + 0.8 * rnd)
                   + u_sway_gust * sin(u_time_s * u_gust_freq + t1.y);
        wp.xy += u_wind_dir * (lean * u_sway_gain * w);
    }

    gl_Position = p3d_ModelViewProjectionMatrix * vec4(wp, 1.0);

    float variant = t1.w;
    v_uv = vec2((variant + p3d_MultiTexCoord0.x) / u_variants,
                p3d_MultiTexCoord0.y);
    v_base_world = t0.xyz;
    v_tint = t1.z;
}
