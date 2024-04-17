from typing import Optional, Tuple, Union

import nibabel as nib
import numpy as np

from niclips.checks import check_4d
from niclips.typing import Coord, NiftiLike

from ._convert import get_fdata
from ._coord import coord2ind


def slice_volume(
    img: NiftiLike,
    coord: Coord = (0.0, 0.0, 0.0),
    axis: int = 0,
    idx: Optional[int] = 0,
) -> np.ndarray:
    """Slice volume at a coordinate along an axis."""
    if img.ndim == 4:
        img = index_img(img, idx=idx)
    data = get_fdata(img)

    if isinstance(img, nib.Nifti1Image):
        coord = coord2ind(img.affine, coord)
    slc = slice_array(data, int(coord[axis]), axis=axis)
    return slc


def index_img(img: NiftiLike, idx: Optional[int] = 0) -> NiftiLike:
    """Index a 4D nifti image. If `idx` is `None`, return the middle volume."""
    check_4d(img)
    if idx is None:
        idx = img.shape[-1] // 2

    # NOTE: We currently assume throughout that 4d nifti can be loaded in memory in
    # full, and that calls to get_fdata() typically return cached data. This
    # dramatically improves access times, compared to interacting with the memmapped
    # dataobj.
    data = get_fdata(img)
    slc = data[..., idx]

    if isinstance(img, nib.Nifti1Image):
        slc_nii = nib.Nifti1Image(slc, affine=img.affine)
    return slc_nii


def crop_middle_third(
    data: np.ndarray, axis: Union[int, Tuple[int, ...]] = 0
) -> np.ndarray:
    """Crop a data array to the middle third along one or axes."""
    if isinstance(axis, int):
        sz = data.shape[axis]
        cropped = slice_array(data, slice(sz // 3, 2 * sz // 3), axis=axis)
    else:
        cropped = data
        for axi in axis:
            cropped = crop_middle_third(cropped, axis=axi)
    return cropped


def slice_array(data: np.ndarray, idx: Union[int, slice], axis: int = 0) -> np.ndarray:
    """Slice a numpy array along an axis.

    `idx` should be an integer or slice object.
    Similar to `np.take()` but faster since it doesn't default to fancy indexing.

    Reference:
        https://stackoverflow.com/a/52378197
    """
    if isinstance(idx, int):
        idx = idx % data.shape[axis]
    slice_tup = axis * (slice(None),) + (idx,)
    return data[slice_tup]
