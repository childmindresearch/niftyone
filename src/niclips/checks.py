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
    rot = img.affine[:3, :3]
    expected = np.abs(np.diag(np.diag(rot)))
    if not np.allclose(rot, expected, rtol=0.1, atol=0.1):
        raise ValueError(f"Expected RAS orientation; got rotation {rot}")


def check_iso_ras(img: nib.nifti1.Nifti1Image) -> None:
    """Check that an image has RAS axis orientation with isotropic voxels."""
    check_ras(img)
    # pixdim = np.diag(img.affine[:3, :3])
    # if not np.allclose(pixdim, pixdim[0]):
    #     raise ValueError(f"Expected isotropic voxels; got pixdim {pixdim}")
