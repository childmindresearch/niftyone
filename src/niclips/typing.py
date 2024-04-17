"""Defined types used in niclips."""

import os
from typing import Tuple, Union

import nibabel as nib
import numpy as np

StrPath = Union[str, os.PathLike]

Coord = Union[Tuple[float, float, float] | np.ndarray]

NiftiLike = Union[nib.nifti1.Nifti1Image, np.ndarray]
