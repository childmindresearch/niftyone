import numpy as np

from niclips.typing import Coord


def apply_affine(affine: np.ndarray, coord: np.ndarray) -> np.ndarray:
    """Apply an affine transformation to an array of 3D coordinates.

    Args:
        affine: affine transform, shape (4, 4)
        coord: coordinate(s), shape (n, 3) or (3,)

    Returns:
        transformed coordinates
    """
    affine = np.asarray(affine)
    coord = np.asarray(coord).astype(affine.dtype)
    assert affine.shape == (4, 4), "Invalid affine"

    singleton = coord.ndim == 1
    if singleton:
        coord = coord.reshape(1, -1)
    assert coord.shape[1] == 3, "Invalid coord"

    coord = np.concatenate([coord, np.ones((len(coord), 1))], axis=1)
    coord = coord @ affine.T
    coord = coord[:, :3]
    if singleton:
        coord = coord.flatten()
    return coord


def coord2ind(affine: np.ndarray, coord: Coord) -> np.ndarray:
    """Transform coordinates to volume indices."""
    ind = apply_affine(np.linalg.inv(affine), np.asarray(coord))
    ind = ind.astype(np.int32)
    return ind


def ind2coord(affine: np.ndarray, ind: np.ndarray) -> np.ndarray:
    """Transform volume indices to coordinates."""
    coord = apply_affine(affine, ind)
    return coord
