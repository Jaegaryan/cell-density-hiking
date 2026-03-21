// Requires cdh.glsl

// SOURCE MACROS MACROS
#define SSBO_PEAKS_COLORS 36
#define SSBO_POINTS_LABELS 37
#define SSBO_COLORS 38
#define SSBO_VISUALIZATION_XYZW 60
#define SSBO_VISUALIZATION_UV 61
#define SSBO_VISUALIZATION_COLORS 62
#define SSBO_VISUALIZATION_SIZES 63

// SOURCE VERSION MACROS TREE CDH VISUALIZE_CELLS
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

layout (binding=SSBO_VISUALIZATION_XYZW, std430) buffer SSBO_visualization_xyzw {  vec4 visualization_xyzw[]; };
layout (binding=SSBO_VISUALIZATION_UV,   std430) buffer SSBO_visualization_uv   { ivec2 visualization_uv[]; };

void main() {
    if (gl_GlobalInvocationID.x < cells.length()) {
        Cell cell = cells[gl_GlobalInvocationID.x];
        Node node = nodes[cell.node];
        uint i = gl_GlobalInvocationID.x * 8;
        uint j = gl_GlobalInvocationID.x * 12;

        visualization_xyzw[i+0] = vec4(node.xyzw.x - node.radius, node.xyzw.y - node.radius, node.xyzw.z - node.radius, 1.0);
        visualization_xyzw[i+1] = vec4(node.xyzw.x + node.radius, node.xyzw.y - node.radius, node.xyzw.z - node.radius, 1.0);
        visualization_xyzw[i+2] = vec4(node.xyzw.x + node.radius, node.xyzw.y + node.radius, node.xyzw.z - node.radius, 1.0);
        visualization_xyzw[i+3] = vec4(node.xyzw.x - node.radius, node.xyzw.y + node.radius, node.xyzw.z - node.radius, 1.0);
        visualization_xyzw[i+4] = vec4(node.xyzw.x - node.radius, node.xyzw.y - node.radius, node.xyzw.z + node.radius, 1.0);
        visualization_xyzw[i+5] = vec4(node.xyzw.x + node.radius, node.xyzw.y - node.radius, node.xyzw.z + node.radius, 1.0);
        visualization_xyzw[i+6] = vec4(node.xyzw.x + node.radius, node.xyzw.y + node.radius, node.xyzw.z + node.radius, 1.0);
        visualization_xyzw[i+7] = vec4(node.xyzw.x - node.radius, node.xyzw.y + node.radius, node.xyzw.z + node.radius, 1.0);
        visualization_uv[j+0] = ivec2(i+0, i+1);
        visualization_uv[j+1] = ivec2(i+1, i+2);
        visualization_uv[j+2] = ivec2(i+2, i+3);
        visualization_uv[j+3] = ivec2(i+3, i+0);
        visualization_uv[j+4] = ivec2(i+4, i+5);
        visualization_uv[j+5] = ivec2(i+5, i+6);
        visualization_uv[j+6] = ivec2(i+6, i+7);
        visualization_uv[j+7] = ivec2(i+7, i+4);
        visualization_uv[j+8] = ivec2(i+0, i+4);
        visualization_uv[j+9] = ivec2(i+1, i+5);
        visualization_uv[j+10] = ivec2(i+2, i+6);
        visualization_uv[j+11] = ivec2(i+3, i+7);
    }
}

// SOURCE VERSION MACROS TREE CDH VISUALIZE_NEIGHBORS
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

layout (binding=SSBO_VISUALIZATION_XYZW,   std430) buffer SSBO_visualization_xyzw   {  vec4 visualization_xyzw[]; };
layout (binding=SSBO_VISUALIZATION_UV,     std430) buffer SSBO_visualization_uv     { ivec2 visualization_uv[]; };

void main() {
    if (gl_GlobalInvocationID.x < cells.length()) {
        Cell cell = cells[gl_GlobalInvocationID.x];
        Node node = nodes[cell.node];
        uint i = gl_GlobalInvocationID.x + cell.neighbors;

        visualization_xyzw[i] = vec4(node.xyzw.x, node.xyzw.y, node.xyzw.z, 1.0);

        for (uint j = 0; j < cell.n_neighbors; j++) {
            Cell neighbor_cell = cells[cells_neighbors[cell.neighbors + j]];
            Node neighbor_node = nodes[neighbor_cell.node];

            visualization_xyzw[i+1+j] = vec4(neighbor_node.xyzw.x, neighbor_node.xyzw.y, neighbor_node.xyzw.z, 1.0);
            visualization_uv[cell.neighbors + j] = ivec2(i, i+1+j);
        }
    }
}

// SOURCE VERSION MACROS TREE CDH VISUALIZE_MAX_NEIGHBOR
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

layout (binding=SSBO_VISUALIZATION_XYZW,   std430) buffer SSBO_visualization_xyzw   {  vec4 visualization_xyzw[]; };
layout (binding=SSBO_VISUALIZATION_UV,     std430) buffer SSBO_visualization_uv     { ivec2 visualization_uv[]; };

void main() {
    if (gl_GlobalInvocationID.x < cells.length()) {
        Cell cell = cells[gl_GlobalInvocationID.x];
        Cell neighbor_cell = cells[cell.max_neighbor];
        Node node = nodes[cell.node];
        Node neighbor_node = nodes[neighbor_cell.node];
        uint i = gl_GlobalInvocationID.x * 2;

        visualization_xyzw[i] = vec4(node.xyzw.x, node.xyzw.y, node.xyzw.z, 1.0);
        visualization_xyzw[i+1] = vec4(neighbor_node.xyzw.x, neighbor_node.xyzw.y, neighbor_node.xyzw.z, 1.0);
        visualization_uv[gl_GlobalInvocationID.x] = ivec2(i, i+1);
    }
}

// SOURCE VERSION MACROS TREE CDH VISUALIZE_PROMINENCE
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

layout (binding=SSBO_VISUALIZATION_XYZW,   std430) buffer SSBO_visualization_xyzw   {  vec4 visualization_xyzw[]; };
layout (binding=SSBO_VISUALIZATION_UV,     std430) buffer SSBO_visualization_uv     { ivec2 visualization_uv[]; };
layout (binding=SSBO_VISUALIZATION_COLORS, std430) buffer SSBO_visualization_colors {  vec4 visualization_colors[]; };

void main() {
    if (gl_GlobalInvocationID.x < peaks.length()) {
        Peak peak = peaks[gl_GlobalInvocationID.x];
        Cell cell = cells[peak.cell];
        Node node = nodes[cell.node];
        uint max_density = (depth+1) * (max_points_per_node+1);
        uint i = gl_GlobalInvocationID.x * 2;
        uint j = gl_GlobalInvocationID.x;

        visualization_xyzw[i+0] = vec4(node.xyzw.x, node.xyzw.y, node.xyzw.z, 1.0);
        visualization_xyzw[i+1] = vec4(node.xyzw.x, node.xyzw.y, node.xyzw.z - float(peak.prominence) / float(max_density), 1.0);
        visualization_uv[j] = ivec2(i, i+1);
        visualization_colors[j] = vec4(1, 0, 0, 1);
    }
}

// SOURCE VERSION MACROS TREE CDH VISUALIZE_PARENTS
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

layout (binding=SSBO_VISUALIZATION_XYZW,   std430) buffer SSBO_visualization_xyzw   {  vec4 visualization_xyzw[]; };
layout (binding=SSBO_VISUALIZATION_UV,     std430) buffer SSBO_visualization_uv     { ivec2 visualization_uv[]; };
layout (binding=SSBO_VISUALIZATION_COLORS, std430) buffer SSBO_visualization_colors {  vec4 visualization_colors[]; };

void main() {
    if (gl_GlobalInvocationID.x < peaks.length()) {
        Peak peak = peaks[gl_GlobalInvocationID.x];
        Cell cell = cells[peak.cell];
        Cell parent_cell = cells[peaks[peak.parent].cell];
        Node node = nodes[cell.node];
        Node parent_node = nodes[parent_cell.node];
        uint i = gl_GlobalInvocationID.x * 2;
        uint j = gl_GlobalInvocationID.x;

        visualization_xyzw[i+0] = vec4(node.xyzw.x, node.xyzw.y, node.xyzw.z, 1.0);
        visualization_xyzw[i+1] = vec4(parent_node.xyzw.x, parent_node.xyzw.y, parent_node.xyzw.z, 1.0);
        visualization_uv[j] = ivec2(i, i+1);
        visualization_colors[i+0] = vec4(1, 1, 1, 1);
        visualization_colors[i+1] = vec4(0, 0, 1, 1);
    }
}

// SOURCE VERSION MACROS TREE CDH VISUALIZE_CLUSTERS_CELLS
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

layout (binding=SSBO_PEAKS_COLORS,         std430) buffer SSBO_peak_colors          { vec4 peak_colors[]; };
layout (binding=SSBO_VISUALIZATION_COLORS, std430) buffer SSBO_visualization_colors { vec4 visualization_colors[]; };

void main() {
    if (gl_GlobalInvocationID.x < cells.length()) {
        Cell cell = cells[gl_GlobalInvocationID.x];
        vec4 color = peak_colors[cell.cluster];
        uint i = gl_GlobalInvocationID.x * 12;

        visualization_colors[i+0] = color;
        visualization_colors[i+1] = color;
        visualization_colors[i+2] = color;
        visualization_colors[i+3] = color;
        visualization_colors[i+4] = color;
        visualization_colors[i+5] = color;
        visualization_colors[i+6] = color;
        visualization_colors[i+7] = color;
        visualization_colors[i+8] = color;
        visualization_colors[i+9] = color;
        visualization_colors[i+10] = color;
        visualization_colors[i+11] = color;
    }
}

// SOURCE VERSION MACROS TREE CDH VISUALIZE_CLUSTERS_POINTS
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

layout (binding=36, std430) buffer SSBO_peak_colors { vec4 peak_colors[]; };

layout (binding=62, std430) buffer SSBO_visualization_colors { vec4 visualization_colors[]; };

void main() {
    if (gl_GlobalInvocationID.x < n_points) {
        vec4 color = peak_colors[points_cluster[gl_GlobalInvocationID.x]];
        uint i = gl_GlobalInvocationID.x;

        visualization_colors[i] = color;
    }
}

// SOURCE VERSION MACROS TREE CDH VISUALIZE_LABELS_POINTS
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;
layout (binding=SSBO_PEAKS_COLORS,         std430) buffer SSBO_peak_colors          { vec4 peak_colors[]; };
layout (binding=SSBO_POINTS_LABELS,        std430) buffer SSBO_points_labels        { uint points_labels[]; };
layout (binding=SSBO_VISUALIZATION_COLORS, std430) buffer SSBO_visualization_colors { vec4 visualization_colors[]; };

void main() {
    if (gl_GlobalInvocationID.x < n_points) {
        vec4 color = peak_colors[points_labels[gl_GlobalInvocationID.x]];
        uint i = gl_GlobalInvocationID.x;

        visualization_colors[i] = color;
    }
}

// SOURCE VERSION MACROS TREE CDH VISUALIZE_NEIGHBORS_COLORS
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

layout (binding=SSBO_COLORS,         std430) buffer SSBO_colors          { vec4 colors[]; };
layout (binding=SSBO_VISUALIZATION_COLORS, std430) buffer SSBO_visualization_colors { vec4 visualization_colors[]; };

void main() {
    if (gl_GlobalInvocationID.x < cells.length()) {
        Cell cell = cells[gl_GlobalInvocationID.x];

        for (uint j = 0; j < cell.n_neighbors; j++) {
            visualization_colors[cell.neighbors + j] = colors[0];
        }
    }
}

// SOURCE VERSION MACROS CDH VISUALIZE_MAX_NEIGHBOR_COLORS
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

layout (binding=SSBO_COLORS,         std430) buffer SSBO_colors          { vec4 colors[]; };
layout (binding=SSBO_VISUALIZATION_COLORS, std430) buffer SSBO_visualization_colors { vec4 visualization_colors[]; };

void main() {
    if (gl_GlobalInvocationID.x < cells.length()) {
        visualization_colors[gl_GlobalInvocationID.x] = colors[1];
    }
}
