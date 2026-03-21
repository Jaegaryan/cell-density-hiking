// Requires chd.glsl

// SOURCE VERSION MACROS BLELLOCH TREE SPLIT
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

void main() {
    uint i_parent = level_a + gl_GlobalInvocationID.x;
    if (i_parent < level_b) {
        Node parent = nodes[i_parent];
        if (parent.count > max_points_per_node) {
            // set childs of parent
            nodes[i_parent].childs = level_b + scan[gl_GlobalInvocationID.x] * 8;

            // create child nodes
            float child_radius = parent.radius / 2;
            for (uint orthant = 0; orthant < 8; orthant++) {
                // new child
                Node child;
                float sign_x = ((orthant & 1u) == 0) ? 1.0 : -1.0;
                float sign_y = ((orthant & 2u) == 0) ? 1.0 : -1.0;
                float sign_z = ((orthant & 4u) == 0) ? 1.0 : -1.0;
                child.xyzw = parent.xyzw + vec4(sign_x, sign_y, sign_z, 0.0) * child_radius;
                child.radius = child_radius;
                child.parent = i_parent;
                child.childs = 0;
                child.count = 0;

                // copy to ssbo
                nodes[nodes[i_parent].childs + orthant] = child;
            }
        }
    }
}

// SOURCE VERSION MACROS TREE NEIGHBORS
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

uvec4 get_outside_neighbors(uint neighbor, uvec4 offset) {
    // no neighbor
    if (neighbor == 0) {
        return uvec4(0);
    }
    uint childs = nodes[neighbor].childs;
    // neighbor w/ childs
    if (childs > 0) {
        return childs + offset;
    }
    // neighbor w/o childs
    return uvec4(neighbor);
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
        Neighbors neighbors_4;
        Neighbors neighbors_5;
        Neighbors neighbors_6;
        Neighbors neighbors_7;

        // inside neighbors
        neighbors_0.north = childs_node + 4;
        neighbors_0.west  = childs_node + 1;
        neighbors_0.down  = childs_node + 2;
        neighbors_1.east  = childs_node + 0;
        neighbors_1.north = childs_node + 5;
        neighbors_1.down  = childs_node + 3;
        neighbors_2.north = childs_node + 6;
        neighbors_2.west  = childs_node + 3;
        neighbors_2.up    = childs_node + 0;
        neighbors_3.north = childs_node + 7;
        neighbors_3.east  = childs_node + 2;
        neighbors_3.up    = childs_node + 1;
        neighbors_4.south = childs_node + 0;
        neighbors_4.west  = childs_node + 5;
        neighbors_4.down  = childs_node + 6;
        neighbors_5.south = childs_node + 1;
        neighbors_5.east  = childs_node + 4;
        neighbors_5.down  = childs_node + 7;
        neighbors_6.west  = childs_node + 7;
        neighbors_6.south = childs_node + 2;
        neighbors_6.up    = childs_node + 4;
        neighbors_7.south = childs_node + 3;
        neighbors_7.east  = childs_node + 6;
        neighbors_7.up    = childs_node + 5;

        // outside neighbors
        Neighbors neighbors_node = neighbors[i_node];
        //north
        uvec4 north = get_outside_neighbors(neighbors_node.north, uvec4(0, 2, 1, 3));
        neighbors_4.north = north.x;
        neighbors_6.north = north.y;
        neighbors_5.north = north.z;
        neighbors_7.north = north.w;
        // east
        uvec4 east = get_outside_neighbors(neighbors_node.east, uvec4(1, 3, 5, 7));
        neighbors_0.east = east.x;
        neighbors_2.east = east.y;
        neighbors_4.east = east.z;
        neighbors_6.east = east.w;
        //south
        uvec4 south = get_outside_neighbors(neighbors_node.south, uvec4(4, 5, 6, 7));
        neighbors_0.south = south.x;
        neighbors_1.south = south.y;
        neighbors_2.south = south.z;
        neighbors_3.south = south.w;
        // west
        uvec4 west = get_outside_neighbors(neighbors_node.west, uvec4(0, 2, 4, 6));
        neighbors_1.west = west.x;
        neighbors_3.west = west.y;
        neighbors_5.west = west.z;
        neighbors_7.west = west.w;
        // up
        uvec4 up = get_outside_neighbors(neighbors_node.up, uvec4(2, 3, 6, 7));
        neighbors_0.up = up.x;
        neighbors_1.up = up.y;
        neighbors_4.up = up.z;
        neighbors_5.up = up.w;
        // down
        uvec4 down = get_outside_neighbors(neighbors_node.down, uvec4(0, 1, 4, 5));
        neighbors_2.down = down.x;
        neighbors_3.down = down.y;
        neighbors_6.down = down.z;
        neighbors_7.down = down.w;

        // copy to ssbo
        neighbors[childs_node + 0] = neighbors_0;
        neighbors[childs_node + 1] = neighbors_1;
        neighbors[childs_node + 2] = neighbors_2;
        neighbors[childs_node + 3] = neighbors_3;
        neighbors[childs_node + 4] = neighbors_4;
        neighbors[childs_node + 5] = neighbors_5;
        neighbors[childs_node + 6] = neighbors_6;
        neighbors[childs_node + 7] = neighbors_7;
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
            uint orthant = uint(point.x < node.xyzw.x) + uint(point.y < node.xyzw.y) * 2u + uint(point.z < node.xyzw.z) * 4u;
            uint new_i_node = node.childs + orthant; 
            atomicAdd(nodes[new_i_node].count, 1u);
            points_node[gl_GlobalInvocationID.x] = new_i_node;
        }
    }
}

// SOURCE VERSION MACROS BLELLOCH TREE PARAMETERS
layout (local_size_x=1, local_size_y=1, local_size_z=1) in;

void main() {
    level_a = level_b;
    level_b += sum * 8;

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

void traverse_neighbor(uint neighbor, uvec4 offset, inout uint n_neighbors) {
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
            stack[++i] = childs + offset.z;
            stack[++i] = childs + offset.w;
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
        if (cell_neighbors.north > 0) { traverse_neighbor(cell_neighbors.north, uvec4(0, 1, 2, 3), n_neighbors); }
        if (cell_neighbors.east  > 0) { traverse_neighbor(cell_neighbors.east,  uvec4(1, 3, 5, 7), n_neighbors); }
        if (cell_neighbors.south > 0) { traverse_neighbor(cell_neighbors.south, uvec4(4, 5, 6, 7), n_neighbors); }
        if (cell_neighbors.west  > 0) { traverse_neighbor(cell_neighbors.west,  uvec4(0, 2, 4, 6), n_neighbors); }
        if (cell_neighbors.up    > 0) { traverse_neighbor(cell_neighbors.up,    uvec4(2, 3, 6, 7), n_neighbors); }
        if (cell_neighbors.down  > 0) { traverse_neighbor(cell_neighbors.down,  uvec4(0, 1, 4, 5), n_neighbors); }

        cells[gl_GlobalInvocationID.x].n_neighbors = n_neighbors;
        scan[gl_GlobalInvocationID.x] = n_neighbors;
    }
}

// SOURCE VERSION MACROS BLELLOCH TREE CDH COMPACT_NEIGHBORS
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

void traverse_neighbor(uint neighbor, uvec4 offset, inout uint i_neighbors) {
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
            stack[++i] = childs + offset.z;
            stack[++i] = childs + offset.w;
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
        if (cell_neighbors.north > 0) { traverse_neighbor(cell_neighbors.north, uvec4(0, 1, 2, 3), i_neighbors); }
        if (cell_neighbors.east  > 0) { traverse_neighbor(cell_neighbors.east,  uvec4(1, 3, 5, 7), i_neighbors); }
        if (cell_neighbors.south > 0) { traverse_neighbor(cell_neighbors.south, uvec4(4, 5, 6, 7), i_neighbors); }
        if (cell_neighbors.west  > 0) { traverse_neighbor(cell_neighbors.west,  uvec4(0, 2, 4, 6), i_neighbors); }
        if (cell_neighbors.up    > 0) { traverse_neighbor(cell_neighbors.up,    uvec4(2, 3, 6, 7), i_neighbors); }
        if (cell_neighbors.down  > 0) { traverse_neighbor(cell_neighbors.down,  uvec4(0, 1, 4, 5), i_neighbors); }

        cells[gl_GlobalInvocationID.x].neighbors = scan[gl_GlobalInvocationID.x];
    }
}
