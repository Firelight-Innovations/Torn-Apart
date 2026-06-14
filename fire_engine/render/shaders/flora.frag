#version 330 core
// flora.frag — flower sprites AND the tree/bush impostor fragment stage
// (tree_renderer.py uses this source verbatim for the far-LOD billboards).
//
// grass.frag's contract with two flora generalisations: the albedo comes
// from a sprite ATLAS (v_uv already remapped per variant by flora.vert /
// tree_impostor.vert), and the cascade sample height above the base is a
// per-kind uniform (0.5 m for a flower, ~45% of max height for a tree
// canopy) instead of grass's hard-coded offset.  Lighting itself is the
// shared lit_surface.glsl contract — cascades, cross-fade, refinement,
// AO, fog, HDR finish — identical to terrain.
#define LIT_REFINE 1
//#include "lit_surface.glsl"

in vec2  v_uv;
in vec3  v_base_world;
in float v_tint;

out vec4 frag_color;

uniform sampler2D u_sprite;       // flora sprite atlas (alpha cutout)
uniform float u_light_offset_m;   // cascade sample height above the base

void main() {
    vec4 albedo = texture(u_sprite, v_uv);
    if (albedo.a < 0.5) discard;      // binary cutout — no blending/sorting

    vec3 base = pow(albedo.rgb, vec3(2.2)) * v_tint;

    // Light at the plant base, snapped to the same light-pixel grid as the
    // terrain, sampled at the kind's canopy height above the ground.
    vec3 wq = litQuantPos(v_base_world, u_quant_m);
    vec3 probe = wq + vec3(0.0, 0.0, u_light_offset_m);
    vec3 radiance, vis;
    float occ;
    sampleCascades(probe, radiance, vis, occ);

    // Re-resolve celestial shadow edges in the penumbra band (single-ray —
    // sprite cutouts are too high-frequency for the 4-ray soft gradient).
    if (u_refine > 0.5) {
        if (u_sun_dir.z > -0.05 && vis.r > 0.02 && vis.r < 0.98)
            vis.r = refineVis(probe, u_sun_dir);
        if (u_moon_dir.z > -0.05 && vis.g > 0.02 && vis.g < 0.98 &&
            dot(u_moon_radiance, vec3(1.0)) > 1e-4)
            vis.g = refineVis(probe, u_moon_dir);
    }

    // Sprites are vertical: Lambert against straight-up normals.
    vec3 direct = u_sun_radiance  * (vis.r * max(u_sun_dir.z,  0.0))
                + u_moon_radiance * (vis.g * max(u_moon_dir.z, 0.0));

    float ao = litAo(occ, occ);       // single-tap AO (billboard geometry)
    vec3 hdr = base * (direct + radiance * ao);

    float dist = length(v_base_world - u_cam_pos);
    frag_color = vec4(litFinish(litFog(hdr, dist)), 1.0);
}
