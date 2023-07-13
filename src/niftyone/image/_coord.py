import numpy as np
import nibabel as nib
from scipy import ndimage

from niftyone.checks import check_3d_4d

from ._convert import get_fdata


def apply_affine(affine: np.ndarray, coord: np.ndarray) -> np.ndarray:
    """
    Apply an affine transformation to an array of 3D coordinates

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


def coord2ind(affine: np.ndarray, coord: np.ndarray) -> np.ndarray:
    """
    Transform coordinates to volume indices.
    """
    ind = apply_affine(np.linalg.inv(affine), coord)
    ind = ind.astype(np.int32)
    return ind


def ind2coord(affine: np.ndarray, ind: np.ndarray) -> np.ndarray:
    """
    Transform volume indices to coordinates.
    """
    coord = apply_affine(affine, ind)
    return coord


def center_of_mass(img: nib.nifti1.Nifti1Image, idx: int = 0, mask: bool = False) -> np.ndarray:
    """
    Find the coordinate for the image center of mass.
    """
    check_3d_4d(img)
    data = get_fdata(img)
    if data.ndim == 4:
        data = data[..., idx]
    data = np.maximum(data, 0.0)

    # Find the center wrt a rough mask mask not the raw values
    if mask:
        data = data > data.mean()

    centroid = ndimage.center_of_mass(data)
    centroid = ind2coord(img.affine, centroid)
    return centroid


def peak_of_mass(img: nib.nifti1.Nifti1Image, idx: int = 0, mask: bool = False) -> np.ndarray:
    """
    Find the coordinate for the "peak" of image mass.

    First identifies the peak in the Z axis, and then finds the X and Y axis peaks
    constrained to that slice. The goal is basically to find the planes passing through
    the widest parts of the brain. The peak coordinate is the intersection of those
    planes.

    Assumes the axes are ordered XYZ[T].
    """
    check_3d_4d(img)
    data = get_fdata(img)
    if data.ndim == 4:
        data = data[..., idx]
    data = np.maximum(data, 0.0)

    # Find the peak wrt a rough mask mask not the raw values
    if mask:
        data = data > data.mean()

    # Find the peak for z axis first, then x and y peak restricted to that slice. The
    # unconstrained y peak might not be where we want due to the presence of neck.
    zidx = np.argmax(np.sum(data, axis=(0, 1)))
    data = data[..., zidx]
    xidx = np.argmax(np.sum(data, axis=1))
    yidx = np.argmax(np.sum(data, axis=0))
    centroid = ind2coord(img.affine, (xidx, yidx, zidx))
    return centroid
