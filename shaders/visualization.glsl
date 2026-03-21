// SOURCE VERSION
#version 450 core

// SOURCE MACROS
#define UBO_CAMERA 0
#define UBO_PARAMETERS 1
#define SSBO_VISUALIZATION_XYZW 60
#define SSBO_VISUALIZATION_UV 61
#define SSBO_VISUALIZATION_COLORS 62
#define SSBO_VISUALIZATION_SIZES 63

// SOURCE BUFFERS
layout (binding=UBO_CAMERA, std140) uniform UBO_Camera {
    mat4 model;
    mat4 view;
    mat4 projection;
};

layout (binding=UBO_PARAMETERS, std140) uniform UBO_parameters {
    float scale_x;
    float scale_y;
    float scale_z;
    float color_alpha;
    float point_size;
};

layout (binding=SSBO_VISUALIZATION_XYZW,   std430) readonly buffer SSBO_xyzw   {  vec4 xyzw[]; };
layout (binding=SSBO_VISUALIZATION_UV,     std430) readonly buffer SSBO_uv     {  uint uv[]; };
layout (binding=SSBO_VISUALIZATION_COLORS, std430) readonly buffer SSBO_colors {  vec4 colors[]; };
layout (binding=SSBO_VISUALIZATION_SIZES,  std430) readonly buffer SSBO_sizes  { float sizes[]; };


// SOURCE VERSION MACROS BUFFERS VERTEX_POINTS
flat out vec4 v_color;

void main() {
    gl_Position = projection * view * (xyzw[gl_VertexID] * vec4(scale_x, scale_y, scale_z, 1));
    gl_PointSize = sizes[gl_VertexID] * point_size / gl_Position.z;
    v_color = colors[gl_VertexID];
    v_color.w = color_alpha;
}

// SOURCE VERSION MACROS BUFFERS VERTEX_LINES
flat out vec4 v_color;

void main() {
    gl_Position = projection * view * (xyzw[uv[2 * gl_InstanceID + gl_VertexID]] * vec4(scale_x, scale_y, scale_z, 1));
    v_color = colors[gl_InstanceID];  // color per line
    v_color.w = color_alpha;
}

// SOURCE VERSION FRAGMENT
flat in vec4 v_color;
out vec4 f_color;

void main() {
    f_color = v_color;
}

// SOURCE VERSION MACROS BUFFERS VERTEX_GRADIENT
out vec4 v_color;

void main() {
    gl_Position = projection * view * (xyzw[uv[2 * gl_InstanceID + gl_VertexID]] * vec4(scale_x, scale_y, scale_z, 1));
    v_color = colors[2 * gl_InstanceID + gl_VertexID];  // color per vertex
    v_color.w = color_alpha;
}

// SOURCE VERSION FRAGMENT_GRADIENT
in vec4 v_color;
out vec4 f_color;

void main() {
    f_color = v_color;
}
