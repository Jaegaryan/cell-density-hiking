import numpy as np
from numba import jit


@jit(cache=True)
def rotation_matrix(a, v):
    """ Transposed rotation matrix
    Args:
        a: angle
        v: vector
        """
    x, y, z = v
    s, c = np.sin(a), np.cos(a)

    return np.array([
        [c + x ** 2 * (1 - c),     y * x * (1 - c) + z * s,  z * x * (1 - c) - y * s,  0],
        [x * y * (1 - c) - z * s,  c + y ** 2 * (1 - c),     z * y * (1 - c) + x * s,  0],
        [x * z * (1 - c) + y * s,  y * z * (1 - c) - x * s,  c + z ** 2 * (1 - c),     0],
        [0., 0, 0, 1],
    ], dtype=np.float32)


@jit(cache=True)
def scale_matrix(v):
    x, y, z = v

    return np.array([
        [x, 0, 0, 0],
        [0, y, 0, 0],
        [0, 0, z, 0],
        [0., 0, 0, 1],
    ], dtype=np.float32)


@jit(cache=True)
def translation_matrix(v):
    """ Transposed translation matrix """
    x, y, z = v

    return np.array([
        [1., 0, 0, 0],
        [0., 1, 0, 0],
        [0., 0, 1, 0],
        [x, y, z, 1],
    ], dtype=np.float32)


@jit(cache=True)
def projection_matrix_perspective(fov, aspect, near, far):
    """
    Args:
        fov: float
            FOV in radians
        aspect: float
            height / width
        near: float
            distance to near plane
        far: float
            distance to far plane

    Returns: projection matrix
    """
    s = 1. / np.tan(fov/2)

    return np.array([[s,  0,          0,                             0],
                     [0., s / aspect, 0,                             0],
                     [0., 0,          (far + near) / (near - far),  -1],
                     [0., 0,          2 * far * near / (near - far), 0]], dtype=np.float32)


# @jit(cache=True)
# def projection_matrix_orthographic(aspect, near, far):
#     top = 1
#     bottom = -1
#     right = 1 / aspect
#     left = -1 / aspect
#
#     return np.array([
#         [2 / (right - left),               0,                                0,                            0],
#         [0,                                2 / (top - bottom),               0,                            0],
#         [0,                                0,                                -2 / (far - near),             0],
#         [-(right + left) / (right - left), -(top + bottom) / (top - bottom), -(far + near) / (far - near), 1],
#     ], dtype=np.float32).T


@jit(cache=True)
def normalize(v):
    return v / np.sqrt(np.sum(v ** 2))


@jit(cache=True)
def orthonormalize(x, y, z):
    y = normalize(y)
    x = normalize(np.cross(y, z))
    z = normalize(np.cross(x, y))

    return np.array([[*x], [*y], [*z]], dtype=np.float32)


@jit(cache=True)
def look_at_matrix(position, target, up):
    """ OpenGL style look_at_matrix """
    d = position - target  # opposite viewing direction
    d /= np.sqrt(np.sum(d ** 2))

    r = np.cross(up, d)
    r /= np.sqrt(np.sum(r ** 2))

    u = np.cross(d, r)
    u /= np.sqrt(np.sum(u ** 2))

    p = position

    return np.dot(
        np.array([
            [*r, 0],
            [*u, 0],
            [*d, 0],
            [0., 0, 0, 1]
        ], dtype=np.float32),
        np.array([
            [1, 0, 0, -p[0]],
            [0, 1, 0, -p[1]],
            [0, 0, 1, -p[2]],
            [0., 0, 0, 1],
        ], dtype=np.float32))


@jit(cache=True)
def view_matrix_orbit(target, radius, basis):
    """ look_at_matrix but with different args
    Args:
        target: array [3]
            point to look at
        radius: float32
            distance from target to camera
        basis: array [3, 3]
            orthonormal basis of the camera

    Returns:
        view matrix
    """
    x = basis[0]
    y = basis[1]
    z = basis[2]
    p = target + z * radius

    return np.array([
        [*x, -np.dot(p, x)],
        [*y, -np.dot(p, y)],
        [*z, -np.dot(p, z)],
        [*np.array([0, 0, 0, 1], dtype=np.float32)],
        ], dtype=np.float32)
