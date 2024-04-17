import nibabel as nib
import numpy as np
from scipy import ndimage

from niclips.checks import check_3d_4d
from niclips.typing import NiftiLike

from ._convert import get_fdata
from ._coord import ind2coord
from ._slice import index_img


def center_of_mass(img: NiftiLike, mask: bool = False) -> np.ndarray:
    """Find the coordinate for the image center of mass."""
    check_3d_4d(img)
    if img.ndim == 4:
        img = index_img(img, idx=None)
    data = get_fdata(img)
    data = np.maximum(data, 0.0)

    # Find the center wrt a rough mask mask not the raw values
    if mask:
        data = data > data.mean()

    centroid = ndimage.center_of_mass(data)
    if isinstance(img, nib.Nifti1Image):
        centroid = ind2coord(img.affine, centroid)
    return centroid


def peak_of_mass(img: NiftiLike, mask: bool = False) -> np.ndarray:
    """Find the coordinate for the "peak" of image mass.

    First identifies the peak in the Z axis, and then finds the X and Y axis peaks
    constrained to that slice. The goal is basically to find the planes passing through
    the widest parts of the brain. The peak coordinate is the intersection of those
    planes.

    Assumes the axes are ordered XYZ[T].
    """
    check_3d_4d(img)
    if img.ndim == 4:
        img = index_img(img, idx=None)
    data = get_fdata(img)
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
    centroid = np.asarray((xidx, yidx, zidx))

    if isinstance(img, nib.Nifti1Image):
        centroid = ind2coord(img.affine, centroid)
    return np.asarray(centroid)
