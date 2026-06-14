#version 330 core
// Screen-space crepuscular rays (god rays), radial light-scattering from the
// sun's screen position (Mitchell 2007).  Marches from each pixel toward the
// sun accumulating the bright (sun) signal with distance decay — so shafts
// stream out from the sun and are blocked wherever clouds or terrain are dark
// in the scene buffer (occlusion is automatic, like the lens flare).  Added
// (scaled) at the composite.
uniform sampler2D u_tex;       // HDR scene
uniform vec2  u_sun_screen;    // sun position in UV [0,1]
uniform float u_active;        // 0 when the sun is off-screen / below horizon
uniform int   u_samples;       // march samples toward the sun
uniform float u_density;       // overall ray length (fraction toward the sun)
uniform float u_decay;         // per-step attenuation
uniform float u_threshold;     // isolate the sun (HDR luminance) from bright sky

in vec2 v_uv;
out vec4 frag_color;

void main() {
    if (u_active < 0.5) { frag_color = vec4(0.0, 0.0, 0.0, 1.0); return; }

    vec2 delta = (v_uv - u_sun_screen) * (u_density / float(u_samples));
    vec2 uv = v_uv;
    float illum = 1.0;
    vec3 acc = vec3(0.0);
    for (int i = 0; i < u_samples; ++i) {
        uv -= delta;
        vec3 s = max(texture(u_tex, uv).rgb - u_threshold, 0.0);
        acc += s * illum;
        illum *= u_decay;
    }
    frag_color = vec4(acc / float(u_samples), 1.0);
}
