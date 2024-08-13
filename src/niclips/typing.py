"""Defined types used in niclips."""

import os

import nibabel as nib
import numpy as np

StrPath = str | os.PathLike

Coord = tuple[float, float, float] | np.ndarray

NiftiLike = nib.nifti1.Nifti1Image | np.ndarray
