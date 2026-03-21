// Requires cdh.glsl

// SOURCE MACROS MACROS
#define SSBO_PEAKS_COLORS 36
#define SSBO_POINTS_LABELS 37
#define SSBO_COLORS 38
#define SSBO_VISUALIZATION_XYZW 60
#define SSBO_VISUALIZATION_UV 61
#define SSBO_VISUALIZATION_COLORS 62
#define SSBO_VISUALIZATION_SIZES 63

// SOURCE VERSION MACROS TREE CDH VISUALIZE_POINTS
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

void main() {
    if (gl_GlobalInvocationID.x < n_points) {
        uint node_i = points_node[gl_GlobalInvocationID.x];
        Cell cell = cells[nodes_cells[node_i]];
        uint max_density = (depth+1) * (max_points_per_node+1);
        float z = float(cell.density) / float(max_density);
        uint i = gl_GlobalInvocationID.x;

        vec4 point = points[i];

        points[i] = vec4(point.x, point.y, z, 1.0);
    }
}

// SOURCE VERSION MACROS TREE CDH VISUALIZE_CELLS
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

layout (binding=SSBO_VISUALIZATION_XYZW, std430) buffer SSBO_visualization_xyzw {  vec4 visualization_xyzw[]; };
layout (binding=SSBO_VISUALIZATION_UV,   std430) buffer SSBO_visualization_uv   { ivec2 visualization_uv[]; };

void main() {
    if (gl_GlobalInvocationID.x < cells.length()) {
        Cell cell = cells[gl_GlobalInvocationID.x];
        Node node = nodes[cell.node];
        uint max_density = (depth+1) * (max_points_per_node+1);
        float z = float(cell.density) / float(max_density);
        uint i = gl_GlobalInvocationID.x * 4;

        visualization_xyzw[i+0] = vec4(node.xyzw.x - node.radius, node.xyzw.y - node.radius, z, 1.0);
        visualization_xyzw[i+1] = vec4(node.xyzw.x + node.radius, node.xyzw.y - node.radius, z, 1.0);
        visualization_xyzw[i+2] = vec4(node.xyzw.x + node.radius, node.xyzw.y + node.radius, z, 1.0);
        visualization_xyzw[i+3] = vec4(node.xyzw.x - node.radius, node.xyzw.y + node.radius, z, 1.0);
        visualization_uv[i+0] = ivec2(i+0, i+1);
        visualization_uv[i+1] = ivec2(i+1, i+2);
        visualization_uv[i+2] = ivec2(i+2, i+3);
        visualization_uv[i+3] = ivec2(i+3, i+0);
    }
}

// SOURCE VERSION MACROS TREE CDH VISUALIZE_NEIGHBORS
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

layout (binding=SSBO_VISUALIZATION_XYZW, std430) buffer SSBO_visualization_xyzw {  vec4 visualization_xyzw[]; };
layout (binding=SSBO_VISUALIZATION_UV,   std430) buffer SSBO_visualization_uv   { ivec2 visualization_uv[]; };

void main() {
    if (gl_GlobalInvocationID.x < cells.length()) {
        Cell cell = cells[gl_GlobalInvocationID.x];
        Node node = nodes[cell.node];
        uint max_density = (depth+1) * (max_points_per_node+1);
        float z = float(cell.density) / float(max_density);
        uint i = gl_GlobalInvocationID.x + cell.neighbors;

        visualization_xyzw[i] = vec4(node.xyzw.x, node.xyzw.y, z, 1.0);

        for (uint j = 0; j < cell.n_neighbors; j++) {
            Cell neighbor_cell = cells[cells_neighbors[cell.neighbors + j]];
            Node neighbor_node = nodes[neighbor_cell.node];
            float neighbor_z = float(neighbor_cell.density) / float(max_density);

            visualization_xyzw[i+1+j] = vec4(neighbor_node.xyzw.x, neighbor_node.xyzw.y, neighbor_z, 1.0);
            visualization_uv[cell.neighbors + j] = ivec2(i, i+1+j);
        }
    }
}

// SOURCE VERSION MACROS TREE CDH VISUALIZE_MAX_NEIGHBOR
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

layout (binding=SSBO_VISUALIZATION_XYZW, std430) buffer SSBO_visualization_xyzw {  vec4 visualization_xyzw[]; };
layout (binding=SSBO_VISUALIZATION_UV,   std430) buffer SSBO_visualization_uv   { ivec2 visualization_uv[]; };

void main() {
    if (gl_GlobalInvocationID.x < cells.length()) {
        Cell cell = cells[gl_GlobalInvocationID.x];
        Cell neighbor_cell = cells[cell.max_neighbor];
        Node node = nodes[cell.node];
        Node neighbor_node = nodes[neighbor_cell.node];
        uint max_density = (depth+1) * (max_points_per_node+1);
        float z = float(cell.density) / float(max_density);
        float neighbor_z = float(neighbor_cell.density) / float(max_density);
        uint i = gl_GlobalInvocationID.x * 2;

        visualization_xyzw[i] = vec4(node.xyzw.x, node.xyzw.y, z, 1.0);
        visualization_xyzw[i+1] = vec4(neighbor_node.xyzw.x, neighbor_node.xyzw.y, neighbor_z, 1.0);
        visualization_uv[gl_GlobalInvocationID.x] = ivec2(i, i+1);
    }
}

// SOURCE VERSION MACROS CDH VISUALIZE_HIKE
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

layout (binding=SSBO_PEAKS_COLORS,         std430) buffer SSBO_peak_colors          { vec4 peak_colors[]; };
layout (binding=SSBO_VISUALIZATION_COLORS, std430) buffer SSBO_visualization_colors { vec4 visualization_colors[]; };

void main() {
    if (gl_GlobalInvocationID.x < cells.length()) {
        Cell cell = cells[gl_GlobalInvocationID.x];
        vec4 color = peak_colors[cell.peak];
        uint i = gl_GlobalInvocationID.x * 4;

        visualization_colors[i+0] = color;
        visualization_colors[i+1] = color;
        visualization_colors[i+2] = color;
        visualization_colors[i+3] = color;
    }
}

// SOURCE VERSION MACROS TREE CDH VISUALIZE_EDGES
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

layout (binding=SSBO_PEAKS_COLORS,         std430) buffer SSBO_peak_colors          { vec4 peak_colors[]; };
layout (binding=SSBO_VISUALIZATION_XYZW,   std430) buffer SSBO_visualization_xyzw   { vec4 visualization_xyzw[]; };
layout (binding=SSBO_VISUALIZATION_UV,     std430) buffer SSBO_visualization_uv     { ivec2 visualization_uv[]; };
layout (binding=SSBO_VISUALIZATION_COLORS, std430) buffer SSBO_visualization_colors { vec4 visualization_colors[]; };

void main() {
    if (gl_GlobalInvocationID.x < edges.length()) {
        Edge edge = edges[gl_GlobalInvocationID.x];
        Cell from_cell = cells[peaks[edge.from].cell];
        Cell col_cell = cells[edge.col];
        Cell to_cell = cells[peaks[edge.to].cell];
        Node from_node = nodes[from_cell.node];
        Node col_node = nodes[col_cell.node];
        Node to_node = nodes[to_cell.node];
        uint max_density = (depth+1) * (max_points_per_node+1);
        float from_z = float(from_cell.density) / float(max_density);
        float col_z = float(col_cell.density) / float(max_density);
        float to_z = float(to_cell.density) / float(max_density);
        uint i = gl_GlobalInvocationID.x * 2;
        uint j = gl_GlobalInvocationID.x;

        visualization_xyzw[i+0] = vec4(from_node.xyzw.x, from_node.xyzw.y, from_z, 1.0);
        visualization_xyzw[i+1] = vec4(to_node.xyzw.x, to_node.xyzw.y, to_z, 1.0);
        visualization_uv[j+0] = ivec2(i+0, i+1);
        visualization_colors[j+0] = vec4(col_z, 0, 0, 1);
    }
}

// SOURCE VERSION MACROS TREE CDH VISUALIZE_PROMINENCE
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

layout (binding=SSBO_VISUALIZATION_XYZW,   std430) buffer SSBO_visualization_xyzw   { vec4 visualization_xyzw[]; };
layout (binding=SSBO_VISUALIZATION_UV,     std430) buffer SSBO_visualization_uv     { ivec2 visualization_uv[]; };
layout (binding=SSBO_VISUALIZATION_COLORS, std430) buffer SSBO_visualization_colors { vec4 visualization_colors[]; };

void main() {
    if (gl_GlobalInvocationID.x < peaks.length()) {
        Peak peak = peaks[gl_GlobalInvocationID.x];
        Cell cell = cells[peak.cell];
        Node node = nodes[cell.node];
        uint max_density = (depth+1) * (max_points_per_node+1);
        float z = float(cell.density) / float(max_density);
        float prominence_z = float(cell.density - peak.prominence) / float(max_density);
        uint i = gl_GlobalInvocationID.x * 2;
        uint j = gl_GlobalInvocationID.x;

        visualization_xyzw[i+0] = vec4(node.xyzw.x, node.xyzw.y, z, 1.0);
        visualization_xyzw[i+1] = vec4(node.xyzw.x, node.xyzw.y, prominence_z, 1.0);
        visualization_uv[j] = ivec2(i, i+1);
        visualization_colors[j] = vec4(1, 0, 0, 1);
    }
}

// SOURCE VERSION MACROS TREE CDH VISUALIZE_PARENTS
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

layout (binding=SSBO_VISUALIZATION_XYZW,   std430) buffer SSBO_visualization_xyzw   { vec4 visualization_xyzw[]; };
layout (binding=SSBO_VISUALIZATION_UV,     std430) buffer SSBO_visualization_uv     { ivec2 visualization_uv[]; };
layout (binding=SSBO_VISUALIZATION_COLORS, std430) buffer SSBO_visualization_colors { vec4 visualization_colors[]; };

void main() {
    if (gl_GlobalInvocationID.x < peaks.length()) {
        Peak peak = peaks[gl_GlobalInvocationID.x];
        Cell cell = cells[peak.cell];
        Cell parent_cell = cells[peaks[peak.parent].cell];
        Node node = nodes[cell.node];
        Node parent_node = nodes[parent_cell.node];
        uint max_density = (depth+1) * (max_points_per_node+1);
        float z = float(cell.density) / float(max_density);
        float parent_z = float(parent_cell.density) / float(max_density);
        uint i = gl_GlobalInvocationID.x * 2;
        uint j = gl_GlobalInvocationID.x;

        visualization_xyzw[i+0] = vec4(node.xyzw.x, node.xyzw.y, z, 1.0);
        visualization_xyzw[i+1] = vec4(parent_node.xyzw.x, parent_node.xyzw.y, parent_z, 1.0);
        visualization_uv[j] = ivec2(i, i+1);
        visualization_colors[i+0] = vec4(0, 0, 0, 1);
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
        uint i = gl_GlobalInvocationID.x * 4;

        visualization_colors[i+0] = color;
        visualization_colors[i+1] = color;
        visualization_colors[i+2] = color;
        visualization_colors[i+3] = color;
    }
}

// SOURCE VERSION MACROS TREE CDH VISUALIZE_CLUSTERS_POINTS
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

layout (binding=SSBO_PEAKS_COLORS,         std430) buffer SSBO_peak_colors          { vec4 peak_colors[]; };
layout (binding=SSBO_VISUALIZATION_COLORS, std430) buffer SSBO_visualization_colors { vec4 visualization_colors[]; };

void main() {
    if (gl_GlobalInvocationID.x < points.length()) {
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
