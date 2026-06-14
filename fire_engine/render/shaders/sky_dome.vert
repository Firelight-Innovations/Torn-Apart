
#version 330 core
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
out vec3 v_dir;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    // The dome is camera-centred with no rotation, so the model-space vertex
    // position is exactly the world-space view direction.
    v_dir = p3d_Vertex.xyz;
}
