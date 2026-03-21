// Requires blelloch.glsl

// SOURCE VERSION
#version 450 core

// SOURCE MACROS MACROS
#define SSBO_POINTS 0
#define SSBO_NODES 1
#define SSBO_POINTS_NODE 2
#define SSBO_NEIGHBORS 3
#define SSBO_CELLS 4
#define SSBO_PEAKS 5
#define SSBO_TREE 6
#define SSBO_BORDERS 7
#define SSBO_EDGES 8
#define SSBO_NODES_CELLS 9
#define SSBO_CELLS_NEIGHBORS 10
#define SSBO_POINTS_CLUSTERS 11
#define SSBO_TMP 12
#define SSBO_SORTED 13

// SOURCE TREE
struct Node {
    vec4 xyzw;
    float radius;
    uint parent;
    uint childs;
    uint count;
};
struct Neighbors {
    uint north;
    uint east;
    uint south;
    uint west;
    uint up;
    uint down;
};
layout (binding=SSBO_POINTS,      std430) buffer SSBO_points      { vec4 points[]; };
layout (binding=SSBO_NODES,       std430) buffer SSBO_nodes       { Node nodes[]; };
layout (binding=SSBO_POINTS_NODE, std430) buffer SSBO_points_node { uint points_node[]; };
layout (binding=SSBO_NEIGHBORS,   std430) buffer SSBO_neighbors   { Neighbors neighbors[]; };
layout (binding=SSBO_TREE,        std430) buffer SSBO_tree {
    uint n_points;
    uint max_points_per_node;
    uint level_a;
    uint level_b;
    uint depth;
};

// SOURCE SORT
layout (binding=SSBO_TMP,    std430) buffer SSBO_tmp    { uint tmp[]; };
layout (binding=SSBO_SORTED, std430) buffer SSBO_sorted { uint sorted[]; };


// SOURCE CDH
struct Cell {
    uint node;
    uint density;
    uint neighbors;
    uint n_neighbors;
    uint max_neighbor;
    uint peak;
    uint cluster;
};
struct Peak {
    uint cell;
    uint height;
    uint prominence;
    uint parent;
    uint cluster;
};
struct Edge {
    uint from;  // peak
    uint col;  // cell
    uint col_height;  // cell.density
    uint to;  // peak
};

layout (binding=SSBO_CELLS,           std430) buffer SSBO_cells           { Cell cells[]; };
layout (binding=SSBO_PEAKS,           std430) buffer SSBO_peaks           { Peak peaks[]; };
layout (binding=SSBO_BORDERS,         std430) buffer SSBO_borders         { Edge borders[]; };
layout (binding=SSBO_EDGES,           std430) buffer SSBO_edges           { Edge edges[]; };
layout (binding=SSBO_NODES_CELLS,     std430) buffer SSBO_nodes_cells     { uint nodes_cells[]; };
layout (binding=SSBO_CELLS_NEIGHBORS, std430) buffer SSBO_cells_neighbors { uint cells_neighbors[]; };
layout (binding=SSBO_POINTS_CLUSTERS, std430) buffer SSBO_points_cluster  { uint points_cluster[]; };

// SOURCE VERSION MACROS TREE SETUP_TREE
layout (local_size_x=1, local_size_y=1, local_size_z=1) in;

void main() {
    nodes[0].radius = 1;
    nodes[0].count = n_points;
}

// SOURCE VERSION MACROS BLELLOCH TREE CHECK_SPLIT
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

void main() {
    uint i = level_a + gl_GlobalInvocationID.x;
    uint value = 0;
    if (i < level_b) {
        if (nodes[i].count > max_points_per_node) {
            value = 1;
        };
    };
    scan[gl_GlobalInvocationID.x] = value;
}

// SOURCE VERSION MACROS BLELLOCH TREE SPLIT

// SOURCE VERSION MACROS TREE NEIGHBORS

// SOURCE VERSION MACROS TREE ORTHANT

// SOURCE VERSION MACROS BLELLOCH TREE PARAMETERS


// SOURCE VERSION MACROS BLELLOCH TREE CDH IS_CELL
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

void main() {
    if (gl_GlobalInvocationID.x < nodes.length()) {
        if (nodes[gl_GlobalInvocationID.x].childs == 0) {
            scan[gl_GlobalInvocationID.x] = 1;
        }
    }
}

// SOURCE VERSION MACROS BLELLOCH TREE CDH COMPACT_CELL
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

void main() {
    if (gl_GlobalInvocationID.x < nodes.length()) {
        if (nodes[gl_GlobalInvocationID.x].childs == 0) {
            cells[scan[gl_GlobalInvocationID.x]].node = gl_GlobalInvocationID.x;
        }
    }
}

// SOURCE VERSION MACROS BLELLOCH TREE CDH COUNT_NEIGHBORS

// SOURCE VERSION MACROS BLELLOCH TREE CDH COMPACT_NEIGHBORS

// SOURCE VERSION MACROS BLELLOCH TREE CDH DENSITY
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

void main () {
    if (gl_GlobalInvocationID.x < cells.length()) {
        Node node = nodes[cells[gl_GlobalInvocationID.x].node];

        // calculate level from node radius
        float radius = 1;
        uint level = 0;
        while (radius > node.radius) {
            radius /= 2;
            level += 1;
        }
        // density
        cells[gl_GlobalInvocationID.x].density = level * (max_points_per_node + 1) + node.count;
    }
}

// SOURCE VERSION MACROS BLELLOCH TREE CDH MAX_NEIGHBOR
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

void main() {
    if (gl_GlobalInvocationID.x < cells.length() ) {
        Cell cell = cells[gl_GlobalInvocationID.x];

        uint max_density = cell.density;
        uint max_neighbor = gl_GlobalInvocationID.x;

        for (uint i = cell.neighbors; i < cell.neighbors + cell.n_neighbors; i++) {
            if (cells[cells_neighbors[i]].density > max_density) {
                max_density = cells[cells_neighbors[i]].density;
                max_neighbor = cells_neighbors[i];
            }
        }
        cells[gl_GlobalInvocationID.x].max_neighbor = max_neighbor;

        // is peak if no higher neighbor
        if (max_neighbor == gl_GlobalInvocationID.x) {
            scan[gl_GlobalInvocationID.x] = 1;
        }
    }
}

// SOURCE VERSION MACROS BLELLOCH TREE CDH COMPACT_PEAK
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

void main() {
    if (gl_GlobalInvocationID.x < cells.length() ) {
        // no neighbor has higher density -> peak
        if (cells[gl_GlobalInvocationID.x].max_neighbor == gl_GlobalInvocationID.x) {
            uint peak = scan[gl_GlobalInvocationID.x];
            cells[gl_GlobalInvocationID.x].peak = peak;  // set cell's peak
            peaks[peak].cell = gl_GlobalInvocationID.x;  // set peak's cell
            peaks[peak].height = cells[gl_GlobalInvocationID.x].density;  // set peak's height
        } else {
            cells[gl_GlobalInvocationID.x].peak = 0xFFFFFFFFu;  // max uint = is not a peak
        }
    }
}

// SOURCE VERSION MACROS BLELLOCH TREE CDH HIKE
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

void main() {
    if (gl_GlobalInvocationID.x < cells.length() ) {
        Cell cell = cells[gl_GlobalInvocationID.x];
        while (cell.peak == 0xFFFFFFFF) {  // max uint, setting to peaks.length() does not work due to compiler optimizations
            cell = cells[cell.max_neighbor];
        }
        cells[gl_GlobalInvocationID.x].peak = cell.peak;
    }
}

// SOURCE VERSION MACROS BLELLOCH TREE CDH COUNT_BORDERS
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

void main() {
    if (gl_GlobalInvocationID.x < cells.length() ) {
        Cell cell = cells[gl_GlobalInvocationID.x];
        uint n_borders = 0;
        uint peak_height = peaks[cell.peak].height;

        for (uint i = cell.neighbors; i < cell.neighbors + cell.n_neighbors; i++) {
            uint neighbor_peak = cells[cells_neighbors[i]].peak;
            if (neighbor_peak != cell.peak && peaks[neighbor_peak].height >= peak_height) {
                n_borders++;
            }
        }
        atomicAdd(scan[cell.peak], n_borders);
        cells[gl_GlobalInvocationID.x].cluster = n_borders;
    }
}

// SOURCE VERSION MACROS BLELLOCH TREE CDH BORDERS
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

void main() {
    if (gl_GlobalInvocationID.x < cells.length() ) {
        Cell cell = cells[gl_GlobalInvocationID.x];
        if (cell.cluster > 0) {  // only consider border cells
            // cell.cluster = tmp storage of number of different and higher neighbor peaks of cell
            // peak.parent = tmp storage of counter of different and higher neighbor peaks of peak
            uint i_borders = scan[cell.peak] + atomicAdd(peaks[cell.peak].parent, cell.cluster);
            uint peak_height = peaks[cell.peak].height;

            for (uint i = cell.neighbors; i < cell.neighbors + cell.n_neighbors; i++) {
                uint neighbor_cell = cells_neighbors[i];
                uint neighbor_peak = cells[neighbor_cell].peak;
                if (neighbor_peak != cell.peak && peaks[neighbor_peak].height >= peak_height) {
                    // move cell to neighbor if col_height (density) is lower
                    Edge border;
                    if (cells[neighbor_cell].density < cell.density) {
                        border.from = cell.peak;
                        border.col = neighbor_cell;
                        border.col_height = cells[neighbor_cell].density;
                        border.to = neighbor_peak;
                    } else {
                        border.from = cell.peak;
                        border.col = gl_GlobalInvocationID.x;
                        border.col_height = cell.density;
                        border.to = neighbor_peak;
                    }
                    borders[i_borders++] = border;
                }
            }
        }
    }
}

// SOURCE VERSION MACROS BLELLOCH TREE CDH EDGES
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

void main() {
    if (gl_GlobalInvocationID.x < peaks.length() ) {
        uint offset = scan[gl_GlobalInvocationID.x];
        uint n_borders = peaks[gl_GlobalInvocationID.x].parent;  // tmp storage of n_borders via atomicAdd in borders_source
        uint count = 0;
        for (uint i = 0; i < n_borders; i++) {
            Edge border_i = borders[offset + i];

            // compare each element to the previous ones and set col_height
            uint j;  // so that it can be used outside of the for loop
            for (j = 0; j < count; j++) {
                Edge border_j = borders[offset + j];
                if (border_i.to == border_j.to) {
                    if (border_i.col_height > border_j.col_height) {
                        borders[offset + j] = border_i;
                    }
                    break;
                }
            }
            if (j == count) {  // this means there was no match -> put after last element and count up
                borders[offset + j] = border_i;
                count++;
            }
        }
        // tmp store cause needed in compact_edges_source
        peaks[gl_GlobalInvocationID.x].parent = offset;
        peaks[gl_GlobalInvocationID.x].cluster = count;
        scan[gl_GlobalInvocationID.x] = count;
    }
}

// SOURCE VERSION MACROS BLELLOCH TREE CDH COMPACT_EDGES
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

void main() {
    if (gl_GlobalInvocationID.x < peaks.length()) {
        uint offset = scan[gl_GlobalInvocationID.x];
        uint n_edges = peaks[gl_GlobalInvocationID.x].cluster;
        // copy from borders to edges
        for (uint i = 0; i < n_edges; i++) {
            edges[offset + i] = borders[peaks[gl_GlobalInvocationID.x].parent + i];
        }
        // reset tmp offset storage
        peaks[gl_GlobalInvocationID.x].parent = 0xFFFFFFFF;
        peaks[gl_GlobalInvocationID.x].cluster = gl_GlobalInvocationID.x;
    }
}

// SOURCE VERSION MACROS BLELLOCH CDH SORT COUNT_SORT_EDGES
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

void main() {
    if (gl_GlobalInvocationID.x < edges.length()) {
        tmp[gl_GlobalInvocationID.x] = atomicAdd(scan[edges[gl_GlobalInvocationID.x].col_height], 1);  // store offset
    }
}

// SOURCE VERSION MACROS BLELLOCH CDH SORT SORT_EDGES
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

void main() {
    if (gl_GlobalInvocationID.x < edges.length()) {
        uint i = scan[edges[gl_GlobalInvocationID.x].col_height] + tmp[gl_GlobalInvocationID.x];
        sorted[i] = gl_GlobalInvocationID.x;
    }
}

// SOURCE VERSION MACROS BLELLOCH TREE CDH SORT CLUSTER_PEAKS
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

void main() {
    if (gl_GlobalInvocationID.x < peaks.length()) {
        uint i = gl_GlobalInvocationID.x;
        while (uint(peaks[i].prominence) < tmp[0]) {
            i = peaks[i].parent;
        }
        peaks[gl_GlobalInvocationID.x].cluster = peaks[i].cluster;
    }
}

// SOURCE VERSION MACROS BLELLOCH TREE CDH CLUSTER_CELLS
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

void main() {
    if (gl_GlobalInvocationID.x < cells.length()) {
        cells[gl_GlobalInvocationID.x].cluster = peaks[cells[gl_GlobalInvocationID.x].peak].cluster;
    }
}

// SOURCE VERSION MACROS BLELLOCH TREE CDH CLUSTER_POINTS
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

void main() {
    if (gl_GlobalInvocationID.x < points_node.length()) {
        points_cluster[gl_GlobalInvocationID.x] = cells[nodes_cells[points_node[gl_GlobalInvocationID.x]]].cluster;
    }
}

// SOURCE VERSION MACROS BLELLOCH TREE CDH RESET_CLUSTER_PEAKS
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

void main() {
    if (gl_GlobalInvocationID.x < peaks.length()) {
        peaks[gl_GlobalInvocationID.x].cluster = gl_GlobalInvocationID.x;
    }
}
