#version 330 core
// tree.frag — 3-D tree/bush fragment stage.
//
// flora.frag's cascade/fog/ACES contract, upgraded for REAL geometry:
// Lambert against the mesh's per-face normals (back faces flipped — the
// tree root renders two-sided for the leaf quads), and the radiance
// cascades sampled at the FRAGMENT's own quantised world position instead
// of the plant base — so trunks darken under their canopy and crowns catch
// the sky, the payoff of having normals at all.
in vec2  v_uv;
in vec3  v_normal;
in vec3  v_world;
in vec3  v_base_world;
in float v_tint;

out vec4 frag_color;

uniform sampler2D u_atlas;        // species atlas: bark opaque | leaf cutout

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
    vec4 albedo = texture(u_atlas, v_uv);
    if (albedo.a < 0.5) discard;      // leaf cutout (bark region is opaque)

    vec3 base = pow(albedo.rgb, vec3(2.2)) * v_tint;
    vec3 n = normalize(v_normal);
    if (!gl_FrontFacing) n = -n;      // two-sided leaf quads

    // Light at THIS fragment, snapped to the engine's light-pixel grid.
    vec3 wq = (floor(v_world / u_quant_m) + 0.5) * u_quant_m;
    vec3 radiance, vis;
    sampleCascades(wq, radiance, vis);

    vec3 direct = u_sun_radiance  * (vis.r * max(dot(n, u_sun_dir),  0.0))
                + u_moon_radiance * (vis.g * max(dot(n, u_moon_dir), 0.0));

    vec3 hdr = base * (direct + radiance);

    if (u_fog_enabled > 0.5) {
        float dist = length(v_world - u_cam_pos);
        float w = log(max(dist, u_fog_near) / u_fog_near)
                / log(u_fog_far / u_fog_near);
        vec2 suv = gl_FragCoord.xy / u_viewport;
        vec4 fog = texture(u_fog_integrated, vec3(suv, clamp(w, 0.0, 1.0)));
        hdr = hdr * fog.a + fog.rgb;
    }

    vec3 ldr = acesTonemap(hdr * u_exposure);
    frag_color = vec4(pow(ldr, vec3(1.0 / 2.2)), 1.0);
}
