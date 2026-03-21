from pathlib import Path
from unittest import case

import numpy as np
from sklearn.datasets import make_moons, make_circles, make_blobs


def make_chainlink(n_samples=10000, noise=(0.2, 0.02)):
    n_per_ring = n_samples // 2

    # Ring 1: XY-plane
    theta1 = np.linspace(0, 2 * np.pi, n_per_ring)
    ring1 = np.array([
        np.cos(theta1),
        np.sin(theta1),
        np.zeros(n_per_ring)
    ]).T

    # Ring 2: XZ-plane (shifted by 1 unit in X)
    theta2 = np.linspace(0, 2 * np.pi, n_per_ring)
    ring2 = np.array([
        1 + np.cos(theta2),
        np.zeros(n_per_ring),
        np.sin(theta2)
    ]).T

    ring1 += np.random.normal(scale=noise[0], size=ring1.shape)
    ring2 += np.random.normal(scale=noise[1], size=ring2.shape)
    X = np.vstack([ring1, ring2])

    labels = np.hstack([np.zeros(n_per_ring), np.ones(n_per_ring)])
    return X, labels


def make_crown(n_samples=1000000, n_blobs=7, radius=1, noise_floor=0.1):
    """
    Creates a 3D ring 'chain' with dense Gaussian blobs distributed along the circle.
    """
    # 1. Distribute point counts
    n_chain = int(n_samples * 0.1)
    n_blobs_total = n_samples - n_chain
    n_per_blob = n_blobs_total // n_blobs
    remainder = n_blobs_total % n_blobs

    # 2. Generate the Uniform Ring (The Chain)
    # We sample angles theta from 0 to 2*pi
    theta_chain = np.random.uniform(0, 2 * np.pi, n_chain)
    x_chain = radius * np.cos(theta_chain)
    y_chain = radius * np.sin(theta_chain)
    z_chain = np.zeros(n_chain)

    X_chain = np.stack([x_chain, y_chain, z_chain], axis=1)
    X_chain += np.random.normal(scale=noise_floor, size=X_chain.shape)

    # 3. Generate the Blobs along the Ring
    # We place blob centers at equal angular intervals
    blob_angles = np.linspace(0, 2 * np.pi, n_blobs, endpoint=False)
    blob_angles += np.random.rand(*blob_angles.shape) * 0.02

    X_blobs = []
    labels_list = [np.zeros(n_chain)]  # 0 for the chain

    for i, angle in enumerate(blob_angles):
        # Calculate 3D center of the blob on the ring
        center = [radius * np.cos(angle), radius * np.sin(angle), 0]

        # Determine number of points for this blob
        current_n = n_per_blob + (remainder if i == n_blobs - 1 else 0)

        # Create the blob
        sigma = np.random.uniform(0.02, 0.1)  # Varying density
        blob = np.random.normal(loc=center, scale=sigma, size=(current_n, 3))

        X_blobs.append(blob)
        labels_list.append(np.full(current_n, i + 1))  # 1 to n_blobs

    # 4. Final Assemble
    X = np.vstack([X_chain] + X_blobs).astype('float32')
    labels = np.concatenate(labels_list)

    return X, labels


def get_data(dataset, n_points, noise_fraction=0.01, **kwargs):
    """ creates selected dataset, normalizes to [-1, 1], and changes fraction of points to random noise """

    # 2D
    if dataset == 'two_moons':
        x, y = make_moons(n_samples=n_points, **kwargs)
    elif dataset == 'circles':
        x, y = make_circles(n_samples=n_points, **kwargs)
    elif dataset == 'clusterable_data':
        x = np.load(Path('data', 'HDBSCAN', 'clusterable_data.npy')).astype(np.float32)
        # duplicate points
        x = np.repeat(x, 16, axis=0)
        # add noise
        x += np.random.rand(*x.shape).astype(np.float32) * 0.1
        y = np.zeros(len(x), dtype=np.uint32)
    # 2D/3D
    elif dataset == 'blobs':
        x, y = make_blobs(n_samples=n_points, **kwargs);
    # 3D
    elif dataset == 'chainlink':
        x, y = make_chainlink(n_points, **kwargs)
    elif dataset == 'crown':
        x, y = make_crown(n_points, **kwargs)

    # dtype
    x = x.astype(np.float32)
    y = y.astype(np.uint32)

    # normalize [-1, 1]
    x = ((x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0) + 1e-16) - 0.5) * 2

    # noise
    n_points, d = x.shape
    n_noise = int(n_points * noise_fraction)
    x[np.random.randint(0, n_points, n_noise)] = (np.random.rand(n_noise * d).reshape(-1, d) - 0.5) * 2

    return x, y, n_points
