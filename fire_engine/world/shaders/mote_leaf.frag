#version 330 core
// mote_leaf.frag — wind-blown leaf-litter fragment stage.
//
// Cheap tier of the lit_surface.glsl contract (no LIT_REFINE — tumbling
// leaves are a handful of pixels; the refinement march would never read):
// cascades sampled at the leaf centre on the light-pixel grid so a leaf
// and the ground it falls on share light patches, up-Lambert direct
// (leaves tumble, no stable normal), fog + HDR finish like everything else.
//#include "lit_surface.glsl"

in vec2  v_uv;
in vec3  v_base_world;

out vec4 frag_color;

uniform sampler2D u_leaf_tex;     // leaf_sprite atlas (3 variants in a row)

void main() {
    vec4 albedo = texture(u_leaf_tex, v_uv);
    if (albedo.a < 0.5) discard;      // same alpha-test threshold as grass

    vec3 base = pow(albedo.rgb, vec3(2.2));

    // Light at the leaf centre, snapped to the terrain light-pixel grid.
    vec3 wq = litQuantPos(v_base_world, u_quant_m);
    vec3 radiance, vis;
    float occ;
    sampleCascades(wq, radiance, vis, occ);

    // Leaves tumble (no stable normal); light them like grass — Lambert vs
    // straight-up, which reads fine for a billboarded flutter.
    vec3 direct = u_sun_radiance  * (vis.r * max(u_sun_dir.z,  0.0))
                + u_moon_radiance * (vis.g * max(u_moon_dir.z, 0.0));

    vec3 hdr = base * (direct + radiance * litAo(occ, occ));

    float dist = length(v_base_world - u_cam_pos);
    frag_color = vec4(litFinish(litFog(hdr, dist)), albedo.a);
}
