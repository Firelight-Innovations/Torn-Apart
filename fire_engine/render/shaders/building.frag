#version 330 core
// building.frag — free-form building fragment stage.
//
// The shared lit_surface.glsl contract (cascade sampling, soft-penumbra shadow
// refinement, two-point voxel AO, fog, exposure/tonemap finish) applied to a
// real building mesh — the SAME recipe terrain and trees use, so a building is
// lit identically to the ground it stands on (no double tonemap, no drift).
// Albedo is the procedural plaster wall texture (p3d_Texture0); UVs are in
// meters (per-quad) so the texture tiles ~once per meter on every surface.
#define LIT_REFINE 1
//#include "lit_surface.glsl"

in vec3 v_world;
in vec3 v_normal;
in vec2 v_uv;
in vec4 v_color;

out vec4 frag_color;

uniform sampler2D p3d_Texture0;   // plaster_wall albedo (sRGB)

void main() {
    vec3 base = pow(texture(p3d_Texture0, v_uv).rgb, vec3(2.2));  // → linear
    vec3 n = normalize(v_normal);

    // Light at THIS fragment, snapped to the light-pixel grid with
    // power-of-two LOD by screen footprint (distant walls don't shimmer).
    float dist = length(v_world - u_cam_pos);
    float eff = litQuantSize(dist * u_px_rad);
    vec3 wq = litQuantPos(v_world, eff);
    // Probe hops off the surface along the face normal so the sample reads the
    // air cell adjacent to the wall, not the wall's own (future) occupancy.
    vec3 probe = wq + n * (u_c0_cell_m * 0.75);
    vec3 radiance, vis;
    float occ;
    sampleCascades(probe, radiance, vis, occ);

    // Soft-penumbra refinement in the band only (full-sun/shadow → 1 tap).
    // From the UNQUANTISED surface probe so the gradient is continuous.
    vec3 probeS = v_world + n * (u_c0_cell_m * 0.75);
    if (u_refine > 0.5) {
        if (u_sun_dir.z > -0.05 && vis.r > 0.02 && vis.r < 0.98)
            vis.r = refineVisSoft(probeS, u_sun_dir);
        if (u_moon_dir.z > -0.05 && vis.g > 0.02 && vis.g < 0.98 &&
            dot(u_moon_radiance, vec3(1.0)) > 1e-4)
            vis.g = refineVisSoft(probeS, u_moon_dir);
    }

    vec3 direct = u_sun_radiance  * (vis.r * max(dot(n, u_sun_dir),  0.0))
                + u_moon_radiance * (vis.g * max(dot(n, u_moon_dir), 0.0));

    // Voxel AO: occupancy a little farther out along the normal (terrain idiom).
    vec3 aoR, aoV; float occFar;
    sampleCascades(wq + n * (u_c0_cell_m * 1.6), aoR, aoV, occFar);
    float ao = litAo(occ, occFar);

    vec3 hdr = base * (direct + radiance * ao);

    frag_color = vec4(litFinish(litFog(hdr, dist)), 1.0);
}
