import os
from typing import Union

import nibabel as nib
import numpy as np

StrPath = Union[str, os.PathLike]

NiftiLike = Union[nib.nifti1.Nifti1Image, np.ndarray]
