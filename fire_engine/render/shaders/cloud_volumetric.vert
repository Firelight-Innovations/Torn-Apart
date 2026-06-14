#version 330 core
// Volumetric cloud "dome": same inverted-sphere trick as the sky dome — the
// sphere is camera-centred with no rotation, so the model-space vertex position
// IS the world-space view direction.  The fragment shader raymarches the cloud
// slab along that direction.
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
out vec3 v_dir;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    v_dir = p3d_Vertex.xyz;
}
