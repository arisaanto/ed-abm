"""Analysis helpers for later empirical validation phases."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def compute_kde_grid(
    points_xy: Sequence[Tuple[float, float]],
    grid_extent: Tuple[float, float, float, float],
    resolution: float = 1.0,
    bandwidth: float = 1.5,
):
    """Convert interaction points to a normalized smoothed 2D density grid."""

    xmin, xmax, ymin, ymax = grid_extent
    x_edges = np.arange(xmin, xmax + resolution, resolution)
    y_edges = np.arange(ymin, ymax + resolution, resolution)

    if not points_xy:
        return np.zeros((len(y_edges) - 1, len(x_edges) - 1), dtype=float)

    point_array = np.asarray(points_xy, dtype=float)
    if len(point_array) < 2:
        density = np.zeros_like(
            np.meshgrid(
                x_edges[:-1] + (resolution / 2.0),
                y_edges[:-1] + (resolution / 2.0),
            )[0],
            dtype=float,
        )
        variance = max(bandwidth * bandwidth, 1e-6)
        point_x, point_y = point_array[0]
        mesh_x, mesh_y = np.meshgrid(
            x_edges[:-1] + (resolution / 2.0),
            y_edges[:-1] + (resolution / 2.0),
        )
        squared_distance = ((mesh_x - point_x) ** 2) + ((mesh_y - point_y) ** 2)
        density += np.exp(-0.5 * squared_distance / variance)
        density_sum = density.sum()
        if density_sum > 0:
            density /= density_sum
        return density
    x_values = point_array[:, 0]
    y_values = point_array[:, 1]
    mesh_x, mesh_y = np.meshgrid(
        x_edges[:-1] + (resolution / 2.0),
        y_edges[:-1] + (resolution / 2.0),
    )
    try:
        from scipy.stats import gaussian_kde
    except ModuleNotFoundError:
        density = np.zeros_like(mesh_x, dtype=float)
        variance = max(bandwidth * bandwidth, 1e-6)
        for point_x, point_y in point_array:
            squared_distance = ((mesh_x - point_x) ** 2) + ((mesh_y - point_y) ** 2)
            density += np.exp(-0.5 * squared_distance / variance)
    else:
        coordinates = np.vstack([x_values, y_values])
        scale_reference = max(np.std(x_values), np.std(y_values), 1e-6)
        bandwidth_scale = bandwidth / scale_reference
        try:
            kde = gaussian_kde(coordinates, bw_method=bandwidth_scale)
            density = kde(np.vstack([mesh_x.ravel(), mesh_y.ravel()])).reshape(mesh_x.shape)
        except np.linalg.LinAlgError:
            # Short pilot runs can produce nearly collinear interaction clouds.
            # Fall back to a deterministic Gaussian rasterization with a tiny,
            # reproducible jitter so KDE-based validation remains robust.
            density = np.zeros_like(mesh_x, dtype=float)
            variance = max(bandwidth * bandwidth, 1e-6)
            jitter_rng = np.random.default_rng(7)
            jittered_points = point_array + jitter_rng.normal(
                loc=0.0,
                scale=min(0.05, bandwidth / 10.0),
                size=point_array.shape,
            )
            for point_x, point_y in jittered_points:
                squared_distance = ((mesh_x - point_x) ** 2) + ((mesh_y - point_y) ** 2)
                density += np.exp(-0.5 * squared_distance / variance)

    density_sum = density.sum()
    if density_sum > 0:
        density /= density_sum
    return density


def cosine_similarity(grid_a, grid_b):
    """Flatten two 2D grids and compute cosine similarity."""

    flat_a = np.asarray(grid_a, dtype=float).ravel()
    flat_b = np.asarray(grid_b, dtype=float).ravel()

    denominator = np.linalg.norm(flat_a) * np.linalg.norm(flat_b)
    if denominator == 0.0:
        return 0.0
    return float(np.dot(flat_a, flat_b) / denominator)
