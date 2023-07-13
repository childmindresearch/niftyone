from typing import Any, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np

from ._coord import coord2ind
from ._convert import get_fdata


def slice_volume(
    img: Union[nib.Nifti1Image, np.ndarray],
    affine: Optional[np.ndarray] = None,
    coord: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    axis: int = 0,
    idx: int = 0,
):
    """
    Slice volume at a coordinate along an axis. The affine is optional for nibabel nifti
    images.
    """
    if affine is None:
        affine = img.affine
    # TODO: trying to be more careful about slicing the dataobj wasn't obviously
    # better. Test more thoroughly.
    img = get_fdata(img)
    if img.ndim == 4:
        img = img[..., idx]
    ind = coord2ind(affine, coord)
    slc = slice_array(img, ind[axis], axis=axis)
    return slc


def index_img(img: nib.Nifti1Image, idx: int = 0):
    """
    Index a 4D nifti image.
    """
    assert img.ndim == 4, "expected a 4d image"
    # NOTE: We currently assume throughout that 4d nifti can be loaded in memory in
    # full, and that calls to get_fdata() typically return cached data. This
    # dramatically improves access times, compared to interacting with the memmapped
    # dataobj.
    img_data = img.get_fdata()
    # TODO: what about header? 
    img = nib.Nifti1Image(img_data[..., idx], img.affine)
    return img


def crop_middle_third(data: np.ndarray, axis: Union[int, Tuple[int, ...]] = 0):
    """
    Crop a data array to the middle third along one or axes.
    """
    if isinstance(axis, int):
        sz = data.shape[axis]
        cropped = slice_array(data, slice(sz // 3, 2 * sz // 3), axis=axis)
    else:
        cropped = data
        for axi in axis:
            cropped = crop_middle_third(cropped, axis=axi)
    return cropped


def slice_array(
    data: np.ndarray, idx: Union[int, slice], axis: int = 0
) -> np.ndarray:
    """
    Slice a numpy array along an axis. `idx` should be an integer or slice object.
    Similar to `np.take()` but faster since it doesn't default to fancy indexing.

    Reference:
        https://stackoverflow.com/a/52378197
    """
    if isinstance(idx, int):
        idx = idx % data.ndim
    slice_tup = axis * (slice(None),) + (idx,)
    return data[slice_tup]
