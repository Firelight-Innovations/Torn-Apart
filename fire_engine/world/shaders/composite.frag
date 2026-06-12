#version 330 core
// Final composite pass for the HDR post-processing chain.
//
// Reads the linear-HDR scene buffer (the object shaders already multiplied in
// the auto-exposure value, so bloom — added in a later phase — operates on the
// exposed signal, which is physically where glare happens), optionally adds the
// bloom contribution, then applies the ACES filmic tonemap + sRGB gamma that
// every surface shader used to do internally.  This is now the ONE place the
// scene is tonemapped, so high dynamic range survives all the way here.
uniform sampler2D u_scene;     // linear HDR, auto-exposure already applied
uniform sampler2D u_bloom;     // bloom blur (black until the bloom phase wires it)
uniform float u_bloom_strength;
uniform sampler2D u_flare;     // lens-flare features (black until that phase)
uniform float u_flare_strength;
uniform sampler2D u_godray;    // crepuscular rays (black until that phase)
uniform float u_godray_strength;

in vec2 v_uv;
out vec4 frag_color;

vec3 acesTonemap(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14),
                 0.0, 1.0);
}

// Hue-preserving tonemap.  Plain per-channel ACES bleaches saturated highlights
// toward white (the bright sky around a mid-elevation sun turning flat white):
// each channel saturates independently, so blue's R/G catch up to B.  Blending
// in a version that tonemaps the PEAK channel and keeps the original RGB ratio
// preserves the hue as luminance rises, so a bright blue sky stays blue instead
// of washing out — while the sun disc (R≈G≈B) still reads white.
vec3 tonemapHuePreserve(vec3 c) {
    vec3 perCh = acesTonemap(c);                 // filmic, desaturates highlights
    float peak = max(c.r, max(c.g, c.b));
    vec3 ratio = c / max(peak, 1e-4);
    vec3 hue = ratio * acesTonemap(vec3(peak)).x;  // tonemap peak, keep colour
    return mix(perCh, hue, 0.6);
}

void main() {
    vec3 hdr = texture(u_scene, v_uv).rgb;
    hdr += texture(u_bloom, v_uv).rgb * u_bloom_strength;
    hdr += texture(u_flare, v_uv).rgb * u_flare_strength;
    hdr += texture(u_godray, v_uv).rgb * u_godray_strength;
    vec3 ldr = tonemapHuePreserve(hdr);
    frag_color = vec4(pow(ldr, vec3(1.0 / 2.2)), 1.0);
}
