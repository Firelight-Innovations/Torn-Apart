#version 330 core
// Image-based lens flare (Chapman-style, screen-space).
//
// Reads the linear-HDR scene, isolates the very bright bits (the sun), and
// rebuilds the artifacts a real lens scatters from them: GHOSTS (the source
// reflected/mirrored through the screen centre at several scales, with
// chromatic fringing) and a HALO ring.  Because it reads the *rendered* scene,
// occlusion is automatic — when terrain covers the sun the sun isn't bright in
// the buffer, so the flare vanishes (no separate occlusion test needed).
//
// Output is added (scaled by u_flare_strength) in the composite, then tonemaps
// with everything else.
uniform sampler2D u_tex;       // HDR scene buffer
uniform float u_threshold;     // isolate the sun (HDR luminance)
uniform int   u_ghosts;        // ghost count along the centre axis
uniform float u_dispersal;     // ghost spacing (fraction of the centre vector)
uniform float u_halo_width;    // halo ring radius (UV)
uniform float u_chroma;        // chromatic-aberration spread (UV)

in vec2 v_uv;
out vec4 frag_color;

vec3 bright(vec3 c) {
    return max(c - u_threshold, 0.0);
}

// Sample the bright scene with a per-channel radial offset (chromatic fringe).
vec3 sampleChroma(vec2 uv, vec2 dir) {
    return vec3(
        bright(texture(u_tex, uv + dir).rgb).r,
        bright(texture(u_tex, uv).rgb).g,
        bright(texture(u_tex, uv - dir).rgb).b);
}

// Falloff that fades a sample as it nears the screen edge (lens vignette).
float edgeFade(vec2 uv) {
    float w = length(vec2(0.5) - uv) / length(vec2(0.5));
    return pow(clamp(1.0 - w, 0.0, 1.0), 5.0);
}

void main() {
    // Mirror through the screen centre: ghosts of a top-right sun march toward
    // the bottom-left, the way real lens reflections do.
    vec2 uv = 1.0 - v_uv;
    vec2 center = vec2(0.5);
    vec2 ghostVec = (center - uv) * u_dispersal;
    vec2 chroma = normalize(ghostVec + vec2(1e-5)) * u_chroma;

    vec3 result = vec3(0.0);
    for (int i = 0; i < u_ghosts; ++i) {
        vec2 offs = uv + ghostVec * float(i);
        result += sampleChroma(offs, chroma) * edgeFade(offs);
    }

    // Halo ring: sample at a fixed radius along the centre direction.
    vec2 haloVec = normalize(center - uv) * u_halo_width;
    vec2 huv = uv + haloVec;
    float halo = pow(1.0 - abs(length(center - huv) - u_halo_width)
                     / max(u_halo_width, 1e-3), 6.0);
    result += sampleChroma(huv, chroma) * clamp(halo, 0.0, 1.0);

    frag_color = vec4(result, 1.0);
}
