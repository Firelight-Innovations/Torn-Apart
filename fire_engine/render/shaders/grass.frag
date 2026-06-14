#version 330 core
// grass.frag — instanced grass-tuft fragment stage.
//
// Lighting comes verbatim from lit_surface.glsl (the engine-wide lit-surface
// contract — same cascades, cross-fade, shadow refinement and HDR finish as
// the terrain), sampled at the BLADE BASE on the quantised light-pixel grid
// so a tuft and the ground it stands on share light patches.  Blades are
// vertical and too thin for per-face normals, so direct light is Lambert
// against straight-up and the AO is the cheap single-tap form.
#define LIT_REFINE 1
//#include "lit_surface.glsl"

in vec2  v_uv;
in vec3  v_base_world;
in float v_tint;

out vec4 frag_color;

uniform sampler2D u_tuft;         // grass_tuft alpha-cutout texture

// Cascade sample height above the blade base — clear of the ground voxel so
// the probe reads the air cell the blade actually stands in.
const float GRASS_LIGHT_OFFSET_M = 0.75;

void main() {
    vec4 albedo = texture(u_tuft, v_uv);
    if (albedo.a < 0.5) discard;      // binary cutout — no blending/sorting

    vec3 base = pow(albedo.rgb, vec3(2.2)) * v_tint;

    // Light at the blade base, snapped to the same light-pixel grid as the
    // terrain so a tuft and the ground it stands on share light patches.
    vec3 wq = litQuantPos(v_base_world, u_quant_m);
    vec3 probe = wq + vec3(0.0, 0.0, GRASS_LIGHT_OFFSET_M);
    vec3 radiance, vis;
    float occ;
    sampleCascades(probe, radiance, vis, occ);

    // Re-resolve celestial shadow edges in the penumbra band (single-ray —
    // blade cutouts are too high-frequency for the 4-ray soft gradient to
    // read; this lines the edge up with the terrain's refined edge).
    if (u_refine > 0.5) {
        if (u_sun_dir.z > -0.05 && vis.r > 0.02 && vis.r < 0.98)
            vis.r = refineVis(probe, u_sun_dir);
        if (u_moon_dir.z > -0.05 && vis.g > 0.02 && vis.g < 0.98 &&
            dot(u_moon_radiance, vec3(1.0)) > 1e-4)
            vis.g = refineVis(probe, u_moon_dir);
    }

    // Blades are vertical: Lambert against straight-up normals.
    vec3 direct = u_sun_radiance  * (vis.r * max(u_sun_dir.z,  0.0))
                + u_moon_radiance * (vis.g * max(u_moon_dir.z, 0.0));

    float ao = litAo(occ, occ);       // single-tap AO (thin geometry)
    vec3 hdr = base * (direct + radiance * ao);

    float dist = length(v_base_world - u_cam_pos);
    frag_color = vec4(litFinish(litFog(hdr, dist)), 1.0);
}
