import numpy as np
from numba import jit


@jit(cache=True)
def quaternion_multiply(q, v):
    """ Multiply two quaternions
    Args:
        q: array [4]
        v: array [4]

    Returns:
        array [4]
    """
    qw, qi, qj, qk = q
    vw, vi, vj, vk = v

    return np.array([
        qw * vw - qi * vi - qj * vj - qk * vk,
        qw * vi + qi * vw + qj * vk - qk * vj,
        qw * vj - qi * vk + qj * vw + qk * vi,
        qw * vk + qi * vj - qj * vi + qk * vw,
    ], dtype=np.float32)


@jit(cache=True)
def quaternion_rotate(angle, axis, vectors):
    """ Rotates vectors angle radians around axis.
    Args:
        angle: float
            radian
        axis: array [3]
        vectors: array [N, 3]

    Returns:
        rotated vectors: array [N, 3]

    Notes:
        performance optimizations:
            ``` test code
            from time import time

            vectors = np.eye(3, dtype=np.float32)
            angle = np.pi/2
            axis = np.array([1, 0, 0], dtype=np.float32)

            quaternion_rotate(angle, axis, vectors)  # compile

            s = time()
            for _ in range(int(1e5)):
                quaternion_rotate(angle, axis, vectors)
            print(time() - s)
            ```
            numpy broadcasting: 1e5 -> 4.929916620254517
            njit with python for loop: 1e5 -> 1.072009801864624
            njit with compiled for loop: 1e5 -> 0.6305789947509766
            njit with pre allocated out: 1e5 -> 0.2031843662261963
            (not compiled pre allocated out: 1e5 -> 2.5258078575134277)
    """
    q = np.array([np.cos(angle/2), *(axis * np.sin(angle/2))], dtype=np.float32)
    p = np.array([q[0], *(-q[1:])], dtype=np.float32)  # p = q^-1

    out = np.empty_like(vectors)
    for i, v in enumerate(vectors):
        v = np.array([0, *v], dtype=np.float32)
        out[i] = quaternion_multiply(quaternion_multiply(q, v), p)[1:]

    return out


def main():
    from time import time

    vectors = np.eye(3, dtype=np.float32)  # three vectors for testing
    angle = np.pi/2
    axis = np.array([1, 0, 0], dtype=np.float32)

    quaternion_rotate(angle, axis, vectors)  # compile

    s = time()
    for _ in range(int(1e5)):
        quaternion_rotate(angle, axis, vectors)
    print(time() - s)


if __name__ == '__main__':
    main()
