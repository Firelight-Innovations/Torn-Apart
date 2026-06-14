#version 330 core
// Fullscreen-quad vertex shader for the post-processing chain.
//
// FilterManager builds a screen-spanning card with per-vertex texture
// coordinates in [0,1]; this just passes the UV through.  Shared by every
// post pass (composite, bloom, lens flare, FXAA).
in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;
uniform mat4 p3d_ModelViewProjectionMatrix;

out vec2 v_uv;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    v_uv = p3d_MultiTexCoord0;
}
