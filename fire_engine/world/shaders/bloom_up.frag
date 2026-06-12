#version 330 core
// Bloom upsample (3x3 tent) + add.  Reads the coarser already-upsampled level
// (u_src), tent-filters it up, and adds the same-resolution downsample level
// (u_add).  Chained from the smallest mip back up; the progressive add is what
// gives bloom its smooth, wide, multi-scale glow.
uniform sampler2D u_src;   // coarser upsampled bloom (sampled with bilinear)
uniform sampler2D u_add;   // this level's downsample, added back in

in vec2 v_uv;
out vec4 frag_color;

void main() {
    vec2 t = 1.0 / vec2(textureSize(u_src, 0));
    vec3 s = vec3(0.0);
    s += texture(u_src, v_uv + t * vec2(-1.0,  1.0)).rgb;
    s += texture(u_src, v_uv + t * vec2( 0.0,  1.0)).rgb * 2.0;
    s += texture(u_src, v_uv + t * vec2( 1.0,  1.0)).rgb;
    s += texture(u_src, v_uv + t * vec2(-1.0,  0.0)).rgb * 2.0;
    s += texture(u_src, v_uv).rgb * 4.0;
    s += texture(u_src, v_uv + t * vec2( 1.0,  0.0)).rgb * 2.0;
    s += texture(u_src, v_uv + t * vec2(-1.0, -1.0)).rgb;
    s += texture(u_src, v_uv + t * vec2( 0.0, -1.0)).rgb * 2.0;
    s += texture(u_src, v_uv + t * vec2( 1.0, -1.0)).rgb;
    s *= (1.0 / 16.0);
    frag_color = vec4(texture(u_add, v_uv).rgb + s, 1.0);
}
