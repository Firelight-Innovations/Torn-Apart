#version 330 core
// flora.frag — flowers / bushes / trees fragment stage.
//
// grass.frag's lighting contract verbatim (binary alpha cutout, radiance
// cascades sampled at the plant base on the quantised light-pixel grid,
// froxel fog, ACES + gamma) with two flora generalisations: the albedo
// comes from a sprite ATLAS (v_uv already remapped per variant by
// flora.vert), and the cascade sample height above the base is a per-kind
// uniform (0.5 m for a flower, 3 m for a tree canopy) instead of grass's
// hard-coded 0.75.
in vec2  v_uv;
in vec3  v_base_world;
in float v_tint;

out vec4 frag_color;

uniform sampler2D u_sprite;       // flora sprite atlas (alpha cutout)
uniform float u_light_offset_m;   // cascade sample height above the base

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
    vec4 albedo = texture(u_sprite, v_uv);
    if (albedo.a < 0.5) discard;      // binary cutout — no blending/sorting

    vec3 base = pow(albedo.rgb, vec3(2.2)) * v_tint;

    // Light at the plant base, snapped to the same light-pixel grid as the
    // terrain, sampled at the kind's canopy height above the ground.
    vec3 wq = (floor(v_base_world / u_quant_m) + 0.5) * u_quant_m;
    vec3 radiance, vis;
    sampleCascades(wq + vec3(0.0, 0.0, u_light_offset_m), radiance, vis);

    // Sprites are vertical: Lambert against straight-up normals.
    vec3 direct = u_sun_radiance  * (vis.r * max(u_sun_dir.z,  0.0))
                + u_moon_radiance * (vis.g * max(u_moon_dir.z, 0.0));

    vec3 hdr = base * (direct + radiance);

    if (u_fog_enabled > 0.5) {
        float dist = length(v_base_world - u_cam_pos);
        float w = log(max(dist, u_fog_near) / u_fog_near)
                / log(u_fog_far / u_fog_near);
        vec2 suv = gl_FragCoord.xy / u_viewport;
        vec4 fog = texture(u_fog_integrated, vec3(suv, clamp(w, 0.0, 1.0)));
        hdr = hdr * fog.a + fog.rgb;
    }

    vec3 ldr = acesTonemap(hdr * u_exposure);
    frag_color = vec4(pow(ldr, vec3(1.0 / 2.2)), 1.0);
}
