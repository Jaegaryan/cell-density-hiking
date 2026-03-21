// Requires cdh.glsl

// SOURCE VERSION MACROS BLELLOCH TREE SPLIT
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

void main() {
    uint i_parent = level_a + gl_GlobalInvocationID.x;
    if (i_parent < level_b) {
        Node parent = nodes[i_parent];
        if (parent.count > max_points_per_node) {
            // set childs of parent
            nodes[i_parent].childs = level_b + scan[gl_GlobalInvocationID.x] * 4;

            // create child nodes
            float child_radius = parent.radius / 2;
            for (uint quadrant = 0; quadrant < 4; quadrant++) {
                // new child
                Node child;
                float sign_x = ((quadrant & 1u) == 0) ? 1.0 : -1.0;
                float sign_y = ((quadrant & 2u) == 0) ? 1.0 : -1.0;
                child.xyzw = parent.xyzw + vec4(sign_x, sign_y, 0.0, 0.0) * child_radius;
                child.radius = child_radius;
                child.parent = i_parent;
                child.childs = 0;
                child.count = 0;

                // copy to ssbo
                nodes[nodes[i_parent].childs + quadrant] = child;
            }
        }
    }
}

// SOURCE VERSION MACROS TREE NEIGHBORS
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

uvec2 get_outside_neighbors(uint neighbor, uvec2 offset) {
    // no neighbor
    if (neighbor == 0) {
        return uvec2(0);
    }
    uint childs = nodes[neighbor].childs;
    // neighbor w/ childs
    if (childs > 0) {
        return childs + offset;
    }
    // neighbor w/o childs
    return uvec2(neighbor);
}

void main() {
    uint i_node = level_a + gl_GlobalInvocationID.x;
    uint childs_node = nodes[i_node].childs;
    if (i_node < level_b && childs_node > 0) {

        // neighbors
        Neighbors neighbors_0;
        Neighbors neighbors_1;
        Neighbors neighbors_2;
        Neighbors neighbors_3;

        // inside neighbors
        neighbors_0.down = childs_node + 2;
        neighbors_0.west = childs_node + 1;
        neighbors_1.east = childs_node + 0;
        neighbors_1.down = childs_node + 3;
        neighbors_2.up = childs_node + 0;
        neighbors_2.west = childs_node + 3;
        neighbors_3.up = childs_node + 1;
        neighbors_3.east = childs_node + 2;

        // outside neighbors
        Neighbors neighbors_node = neighbors[i_node];
        // east
        uvec2 east = get_outside_neighbors(neighbors_node.east, uvec2(1, 3));
        neighbors_0.east = east.x;
        neighbors_2.east = east.y;
        // west
        uvec2 west = get_outside_neighbors(neighbors_node.west, uvec2(0, 2));
        neighbors_1.west = west.x;
        neighbors_3.west = west.y;
        // up
        uvec2 up = get_outside_neighbors(neighbors_node.up, uvec2(2, 3));
        neighbors_0.up = up.x;
        neighbors_1.up = up.y;
        // down
        uvec2 down = get_outside_neighbors(neighbors_node.down, uvec2(0, 1));
        neighbors_2.down = down.x;
        neighbors_3.down = down.y;

        // copy to ssbo
        neighbors[childs_node + 0] = neighbors_0;
        neighbors[childs_node + 1] = neighbors_1;
        neighbors[childs_node + 2] = neighbors_2;
        neighbors[childs_node + 3] = neighbors_3;
    }
}

// SOURCE VERSION MACROS TREE ORTHANT
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

void main() {
    if (gl_GlobalInvocationID.x < n_points) {
        vec4 point = points[gl_GlobalInvocationID.x];
        uint i_node = points_node[gl_GlobalInvocationID.x];
        Node node = nodes[i_node];
        if (node.childs > 0) {
            uint quadrant = uint(point.x < node.xyzw.x) + uint(point.y < node.xyzw.y) * 2u;
            uint new_i_node = node.childs + quadrant; 
            atomicAdd(nodes[new_i_node].count, 1u);
            points_node[gl_GlobalInvocationID.x] = new_i_node;
        }
    }
}

// SOURCE VERSION MACROS BLELLOCH TREE PARAMETERS
layout (local_size_x=1, local_size_y=1, local_size_z=1) in;

void main() {
    level_a = level_b;
    level_b += sum * 4;

    depth++;

    // last element in scan for current calculation -> smallest power of two >= number of nodes in level
    scan_length = level_b - level_a;
    scan_length--;
    scan_length |= scan_length >> 1;
    scan_length |= scan_length >> 2;
    scan_length |= scan_length >> 4;
    scan_length |= scan_length >> 8;
    scan_length |= scan_length >> 16;
    scan_length++;
}

// SOURCE VERSION MACROS BLELLOCH TREE CDH COUNT_NEIGHBORS
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

void traverse_neighbor(uint neighbor, uvec2 offset, inout uint n_neighbors) {
    uint stack[32]; // holds node indexes
    int i = 0;      // stack index
    uint childs;    // childs of neighbor

    stack[0] = neighbor;
    while (i >= 0) {
        childs = nodes[stack[i]].childs;
        if (childs > 0) {
            // push childs facing cell to stack
            stack[i] = childs + offset.x;
            stack[++i] = childs + offset.y;
        } else {
            n_neighbors++;
            i--;
        }
    }
}

void main() {
    if (gl_GlobalInvocationID.x < cells.length()) {
        Neighbors cell_neighbors = neighbors[cells[gl_GlobalInvocationID.x].node];

        uint n_neighbors = 0;

        // traverse, excluding edge neighbors which point to root
        if (cell_neighbors.east > 0) { traverse_neighbor(cell_neighbors.east, uvec2(1, 3), n_neighbors); }
        if (cell_neighbors.west > 0) { traverse_neighbor(cell_neighbors.west, uvec2(0, 2), n_neighbors); }
        if (cell_neighbors.up > 0)   { traverse_neighbor(cell_neighbors.up,   uvec2(2, 3), n_neighbors); }
        if (cell_neighbors.down > 0) { traverse_neighbor(cell_neighbors.down, uvec2(0, 1), n_neighbors); }

        cells[gl_GlobalInvocationID.x].n_neighbors = n_neighbors;
        scan[gl_GlobalInvocationID.x] = n_neighbors;
    }
}

// SOURCE VERSION MACROS BLELLOCH TREE CDH COMPACT_NEIGHBORS
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

void traverse_neighbor(uint neighbor, uvec2 offset, inout uint i_neighbors) {
    uint stack[32]; // holds node indexes
    int i = 0;      // stack index
    uint childs;    // childs of neighbor

    stack[0] = neighbor;
    while (i >= 0) {
        childs = nodes[stack[i]].childs;
        if (childs > 0) {
            // push childs facing cell to stack
            stack[i] = childs + offset.x;
            stack[++i] = childs + offset.y;
        } else {
            cells_neighbors[i_neighbors++] = nodes_cells[stack[i]];
            i--;
        }
    }
}

void main() {
    if (gl_GlobalInvocationID.x < cells.length()) {
        Cell cell = cells[gl_GlobalInvocationID.x];
        Neighbors cell_neighbors = neighbors[cell.node];

        uint i_neighbors = scan[gl_GlobalInvocationID.x];

        // traverse, excluding edge neighbors which point to root
        if (cell_neighbors.east > 0) { traverse_neighbor(cell_neighbors.east, uvec2(1, 3), i_neighbors); }
        if (cell_neighbors.west > 0) { traverse_neighbor(cell_neighbors.west, uvec2(0, 2), i_neighbors); }
        if (cell_neighbors.up > 0)   { traverse_neighbor(cell_neighbors.up,   uvec2(2, 3), i_neighbors); }
        if (cell_neighbors.down > 0) { traverse_neighbor(cell_neighbors.down, uvec2(0, 1), i_neighbors); }

        cells[gl_GlobalInvocationID.x].neighbors = scan[gl_GlobalInvocationID.x];
    }
}
