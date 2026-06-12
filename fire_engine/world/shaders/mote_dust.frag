#version 330 core
in vec2  v_uv;
in float v_fade;                  // life × edge fade from the vertex shader

out vec4 frag_color;

uniform sampler2D u_dust_tex;     // dust_mote soft radial alpha
uniform float u_exposure;         // shared lighting exposure (inherited)

// --- froxel fog (inherited from terrain_root) — dust only needs to dim in
//     the distance so far motes don't punch through fog ---------------------
uniform sampler3D u_fog_integrated;
uniform float u_fog_near;
uniform float u_fog_far;
uniform float u_fog_enabled;
uniform vec2  u_viewport;

void main() {
    vec4 speck = texture(u_dust_tex, v_uv);   // RGB warm tint, A soft falloff
    float a = speck.a * v_fade;
    if (a < 0.004) discard;                    // skip the transparent tails

    // Additive: motes glow.  Pre-multiply the warm tint by the falloff so the
    // additive blend (dst += src) adds light at the centre and nothing at the
    // rim.  A modest exposure tie-in keeps them in scene-luminance range.
    vec3 add = speck.rgb * a * (0.06 * clamp(u_exposure, 0.2, 4.0));

    // Fog: scale the additive contribution down with distance so motes fade
    // into the fog instead of floating bright in front of it.  fog.a is the
    // scene-transmittance (1 = clear, →0 = fully fogged), so multiply by it.
    if (u_fog_enabled > 0.5) {
        vec2 suv = gl_FragCoord.xy / u_viewport;
        // Use the far slice's transmittance as a cheap distance-fog dimmer.
        float w = clamp(gl_FragCoord.z, 0.0, 1.0);
        vec4 fog = texture(u_fog_integrated, vec3(suv, w));
        add *= fog.a;
    }

    frag_color = vec4(add, a);     // RGB additive; A used by additive blend
}
