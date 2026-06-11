#version 330 core
in vec3 v_world;
in vec3 v_normal;
in vec2 v_uv;
in vec4 v_color;

out vec4 frag_color;

uniform sampler2D p3d_Texture0;   // albedo
uniform sampler2D p3d_Texture1;   // tangent-space normal map
uniform sampler2D p3d_Texture2;   // emission map (linear HDR/8-bit)

// --- radiance cascades (lighting/gpu.py contract) -----------------------
uniform sampler3D u_c0_radiance;
uniform sampler3D u_c0_vis;       // r sun, g moon, b sky visibility
uniform sampler3D u_c0_geom;      // rgb albedo, a occupancy
uniform sampler3D u_c0_emis;
uniform vec3  u_c0_origin_m;
uniform float u_c0_cell_m;
uniform float u_c0_cells;
uniform sampler3D u_c1_radiance;
uniform sampler3D u_c1_vis;
uniform sampler3D u_c1_geom;
uniform vec3  u_c1_origin_m;
uniform float u_c1_cell_m;
uniform float u_c1_cells;

uniform vec3  u_sun_dir;
uniform vec3  u_sun_radiance;
uniform vec3  u_moon_dir;
uniform vec3  u_moon_radiance;
uniform vec3  u_sky_ambient;
uniform float u_quant_m;          // light-pixel size (0.25 m)
uniform float u_ao_strength;
uniform float u_exposure;
uniform float u_emission_scale;

// --- froxel fog ----------------------------------------------------------
uniform sampler3D u_fog_integrated;  // rgb accumulated light, a transmittance
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

// Sample a cascade triple (radiance, vis, occupancy) with cascade-0 priority.
void sampleCascades(vec3 wp, out vec3 radiance, out vec3 vis, out float occ) {
    vec3 uv0 = c_uv(wp, u_c0_origin_m, u_c0_cell_m, u_c0_cells);
    if (inBox(uv0, 0.02)) {
        radiance = texture(u_c0_radiance, uv0).rgb;
        vis      = texture(u_c0_vis, uv0).rgb;
        occ      = texture(u_c0_geom, uv0).a;
        return;
    }
    vec3 uv1 = c_uv(wp, u_c1_origin_m, u_c1_cell_m, u_c1_cells);
    if (inBox(uv1, 0.01)) {
        radiance = texture(u_c1_radiance, uv1).rgb;
        vis      = texture(u_c1_vis, uv1).rgb;
        occ      = texture(u_c1_geom, uv1).a;
        return;
    }
    radiance = u_sky_ambient * 0.6;       // beyond all cascades: open sky guess
    vis      = vec3(1.0);
    occ      = 0.0;
}

vec3 acesTonemap(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14),
                 0.0, 1.0);
}

void main() {
    // ------------------------------------------------------------------
    // Surface basis + normal map (TBN from the dominant axis, matching the
    // mesher's planar UV projection: X-facingâ†’(Y,Z), Yâ†’(X,Z), Zâ†’(X,Y)).
    // ------------------------------------------------------------------
    vec3 n  = normalize(v_normal);
    vec3 an = abs(n);
    vec3 t, b;
    if (an.x >= an.y && an.x >= an.z)      { t = vec3(0, 1, 0); b = vec3(0, 0, 1); }
    else if (an.y >= an.z)                 { t = vec3(1, 0, 0); b = vec3(0, 0, 1); }
    else                                   { t = vec3(1, 0, 0); b = vec3(0, 1, 0); }
    t = normalize(t - n * dot(n, t));
    b = normalize(b - n * dot(n, b) - t * dot(t, b));
    vec3 nm = texture(p3d_Texture1, v_uv).xyz * 2.0 - 1.0;
    vec3 N  = normalize(t * nm.x + b * nm.y + n * max(nm.z, 0.3));

    // ------------------------------------------------------------------
    // Light sampling â€” positions quantised to the light-pixel grid so the
    // lighting itself is visibly pixelated (2x2x2 light pixels per voxel).
    // ------------------------------------------------------------------
    vec3 wq = (floor(v_world / u_quant_m) + 0.5) * u_quant_m;
    // Shadow/GI probes hop off the surface along the *face* normal.
    vec3 probe   = wq + n * (u_c0_cell_m * 0.75);
    vec3 radiance, vis;
    float occ;
    sampleCascades(probe, radiance, vis, occ);

    // Voxel AO: occupancy a little farther out along the normal + above.
    vec3 aoR, aoV; float occFar;
    sampleCascades(wq + n * (u_c0_cell_m * 1.6), aoR, aoV, occFar);
    float ao = 1.0 - u_ao_strength * clamp(0.5 * occ + 0.7 * occFar, 0.0, 1.0);

    // ------------------------------------------------------------------
    // Compose: direct celestial + flood-fill GI + emission.
    // ------------------------------------------------------------------
    vec3 base = pow(texture(p3d_Texture0, v_uv).rgb, vec3(2.2))
              * v_color.rgb;

    vec3 direct =
        u_sun_radiance  * (vis.r * max(dot(N, u_sun_dir),  0.0)) +
        u_moon_radiance * (vis.g * max(dot(N, u_moon_dir), 0.0));

    vec3 ownEmis = vec3(0.0);
    vec3 uv0 = c_uv(v_world - n * (u_c0_cell_m * 0.25),
                    u_c0_origin_m, u_c0_cell_m, u_c0_cells);
    if (inBox(uv0, 0.0))
        ownEmis = texture(u_c0_emis, uv0).rgb * u_emission_scale;
    vec3 mapEmis = pow(texture(p3d_Texture2, v_uv).rgb, vec3(2.2)) * 4.0;

    vec3 hdr = base * (direct + radiance * ao) + ownEmis + mapEmis;

    // ------------------------------------------------------------------
    // Volumetric fog composite (one tap into the integrated froxels).
    // ------------------------------------------------------------------
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
