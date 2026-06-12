#version 330 core
// tree.frag — 3-D tree/bush fragment stage.
//
// The full lit_surface.glsl contract upgraded for REAL geometry: Lambert
// against the mesh's per-face normals (back faces flipped — the tree root
// renders two-sided for the leaf quads), the radiance cascades sampled at
// the FRAGMENT's own quantised world position instead of the plant base —
// so trunks darken under their canopy and crowns catch the sky — plus the
// terrain-grade extras the old hand-copy lacked: screen-footprint LOD on
// the light-pixel grid (distant canopies don't shimmer), soft-penumbra
// shadow refinement, and two-point voxel AO.
#define LIT_REFINE 1
//#include "lit_surface.glsl"

in vec2  v_uv;
in vec3  v_normal;
in vec3  v_world;
in vec3  v_base_world;
in float v_tint;

out vec4 frag_color;

uniform sampler2D u_atlas;        // species atlas: bark opaque | leaf cutout

void main() {
    vec4 albedo = texture(u_atlas, v_uv);
    if (albedo.a < 0.5) discard;      // leaf cutout (bark region is opaque)

    vec3 base = pow(albedo.rgb, vec3(2.2)) * v_tint;
    vec3 n = normalize(v_normal);
    if (!gl_FrontFacing) n = -n;      // two-sided leaf quads

    // Light at THIS fragment, snapped to the engine's light-pixel grid with
    // power-of-two LOD by screen footprint (litQuantSize) — distant crowns
    // coarsen their light pixels instead of shimmering.
    float dist = length(v_world - u_cam_pos);
    float eff = litQuantSize(dist * u_px_rad);
    vec3 wq = litQuantPos(v_world, eff);
    // Probes hop off the surface along the normal (terrain idiom) — keeps
    // the sample in air once trees occupy the cascade geometry volumes.
    vec3 probe = wq + n * (u_c0_cell_m * 0.75);
    vec3 radiance, vis;
    float occ;
    sampleCascades(probe, radiance, vis, occ);

    // Soft-penumbra refinement in the band (4-ray cone — trunk/canopy
    // surfaces are large and continuous enough for the gradient to read).
    // From the UNQUANTISED probe so the gradient is continuous.
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

    // Voxel AO: occupancy a little farther out along the normal (terrain's
    // two-point form — reads canopy density once trees enter the volumes).
    vec3 aoR, aoV; float occFar;
    sampleCascades(wq + n * (u_c0_cell_m * 1.6), aoR, aoV, occFar);
    float ao = litAo(occ, occFar);

    vec3 hdr = base * (direct + radiance * ao);

    frag_color = vec4(litFinish(litFog(hdr, dist)), 1.0);
}
