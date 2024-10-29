"""Assertion of image parameters."""

import nibabel as nib
import numpy as np

from niclips.typing import NiftiLike


def check_3d(img: NiftiLike) -> None:
    """Check that an image is 3D."""
    if img.ndim != 3:
        raise ValueError(f"Expected 3d image; got shape {img.shape}")


def check_4d(img: NiftiLike) -> None:
    """Check that an image is 4D."""
    if img.ndim != 4:
        raise ValueError(f"Expected 4d image; got shape {img.shape}")


def check_3d_4d(img: NiftiLike) -> None:
    """Check that an image is 3D or 4D."""
    if img.ndim not in {3, 4}:
        raise ValueError(f"Expected 3d or 4d image; got shape {img.shape}")


def check_atmost_4d(img: NiftiLike) -> None:
    """Check that an image is at most 4D."""
    if img.ndim > 4:
        raise ValueError(f"Expected at most 4d image; got shape {img.shape}")


def check_ras(img: nib.nifti1.Nifti1Image) -> None:
    """Check that an image has RAS axis orientation."""
    trgt_ornt = np.array([[0, 1], [1, 1], [2, 1]])
    img_ornt = nib.orientations.io_orientation(img.affine)
    if not np.array_equal(img_ornt, trgt_ornt):
        raise ValueError("Expected RAS orientation")
