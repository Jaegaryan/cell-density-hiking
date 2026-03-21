// SOURCE VERSION
#version 450 core

// SOURCE MACROS
#define SSBO_SCAN 20
#define SSBO_BLELLOCH 21

// SOURCE BLELLOCH
layout (binding=SSBO_SCAN,     std430) buffer SSBO_scan { uint scan[]; };
layout (binding=SSBO_BLELLOCH, std430) buffer SSBO_blelloch {
    uint scan_length;
    uint offset;
    uint sum;
};

// SOURCE VERSION MACROS BLELLOCH BLELLOCH_UP_STEP
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

void main() {
    uint i = (gl_GlobalInvocationID.x + 1) * (offset * 2) - 1;
    scan[i] += scan[i - offset];
}

// SOURCE VERSION MACROS BLELLOCH BLELLOCH_UP_OFFSET
layout (local_size_x=1, local_size_y=1, local_size_z=1) in;

void main() {
    offset *= 2;
}

// SOURCE VERSION MACROS BLELLOCH BLELLOCH_MID_STEP
layout (local_size_x=1, local_size_y=1, local_size_z=1) in;

void main() {
    sum = scan[scan_length - 1];
    scan[scan_length - 1] = 0;
}

// SOURCE VERSION MACROS BLELLOCH BLELLOCH_DOWN_STEP
layout (local_size_x=1024, local_size_y=1, local_size_z=1) in;

void main() {
    uint i = (gl_GlobalInvocationID.x + 1) * (offset * 2) - 1;

    uint left = scan[i - offset];
    scan[i - offset] = scan[i];
    scan[i] += left;
}

// SOURCE VERSION MACROS BLELLOCH BLELLOCH_DOWN_OFFSET
layout (local_size_x=1, local_size_y=1, local_size_z=1) in;

void main() {
    offset /= 2;
}
