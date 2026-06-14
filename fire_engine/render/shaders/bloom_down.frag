#version 330 core
// Bloom downsample (Call of Duty / Jimenez 2014 13-tap).  Each level halves
// resolution.  The first level (u_prefilter=1) also applies the soft-knee
// bright-pass and a Karis luma-weighted average to kill fireflies (single
// ultra-bright pixels that would otherwise flicker as huge blobs).
uniform sampler2D u_tex;
uniform float u_prefilter;   // 1.0 on the first (bright-pass) level only
uniform float u_threshold;   // HDR luminance where bloom starts
uniform float u_knee;        // soft-knee half-width below the threshold

in vec2 v_uv;
out vec4 frag_color;

// Soft-knee bright-pass: smooth ramp from threshold-knee to threshold+knee.
vec3 prefilter(vec3 c) {
    float br = max(c.r, max(c.g, c.b));
    float k = max(u_knee, 1e-4);
    float soft = clamp(br - u_threshold + k, 0.0, 2.0 * k);
    soft = soft * soft / (4.0 * k);
    float contrib = max(soft, br - u_threshold) / max(br, 1e-4);
    return c * contrib;
}

float karis(vec3 c) { return 1.0 / (1.0 + max(c.r, max(c.g, c.b))); }

void main() {
    vec2 t = 1.0 / vec2(textureSize(u_tex, 0));

    vec3 a = texture(u_tex, v_uv + t * vec2(-2.0,  2.0)).rgb;
    vec3 b = texture(u_tex, v_uv + t * vec2( 0.0,  2.0)).rgb;
    vec3 c = texture(u_tex, v_uv + t * vec2( 2.0,  2.0)).rgb;
    vec3 d = texture(u_tex, v_uv + t * vec2(-2.0,  0.0)).rgb;
    vec3 e = texture(u_tex, v_uv).rgb;
    vec3 f = texture(u_tex, v_uv + t * vec2( 2.0,  0.0)).rgb;
    vec3 g = texture(u_tex, v_uv + t * vec2(-2.0, -2.0)).rgb;
    vec3 h = texture(u_tex, v_uv + t * vec2( 0.0, -2.0)).rgb;
    vec3 i = texture(u_tex, v_uv + t * vec2( 2.0, -2.0)).rgb;
    vec3 j = texture(u_tex, v_uv + t * vec2(-1.0,  1.0)).rgb;
    vec3 k = texture(u_tex, v_uv + t * vec2( 1.0,  1.0)).rgb;
    vec3 l = texture(u_tex, v_uv + t * vec2(-1.0, -1.0)).rgb;
    vec3 m = texture(u_tex, v_uv + t * vec2( 1.0, -1.0)).rgb;

    vec3 result;
    if (u_prefilter > 0.5) {
        // Five overlapping 2x2 boxes, Karis-weighted (centre box weight 0.5).
        vec3 b0 = (j + k + l + m) * 0.25;
        vec3 b1 = (a + b + d + e) * 0.25;
        vec3 b2 = (b + c + e + f) * 0.25;
        vec3 b3 = (d + e + g + h) * 0.25;
        vec3 b4 = (e + f + h + i) * 0.25;
        float w0 = karis(b0) * 0.5;
        float w1 = karis(b1) * 0.125;
        float w2 = karis(b2) * 0.125;
        float w3 = karis(b3) * 0.125;
        float w4 = karis(b4) * 0.125;
        float ws = w0 + w1 + w2 + w3 + w4;
        result = (b0 * w0 + b1 * w1 + b2 * w2 + b3 * w3 + b4 * w4)
               / max(ws, 1e-4);
        result = prefilter(result);
    } else {
        result  = e * 0.125;
        result += (a + c + g + i) * 0.03125;
        result += (b + d + f + h) * 0.0625;
        result += (j + k + l + m) * 0.125;
    }
    frag_color = vec4(result, 1.0);
}
