"""Niftyone types."""

import os
from typing import Tuple, Union

import nibabel as nib
import numpy as np

StrPath = Union[str, os.PathLike]

Coord = Tuple[float, float, float]

NiftiLike = Union[nib.nifti1.Nifti1Image, np.ndarray]
