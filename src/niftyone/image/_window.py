"""
Utilities for normalizing value ranges.
"""

from typing import NamedTuple, Optional

import numpy as np

from niftyone.checks import check_3d_4d
from niftyone.typing import NiftiLike

from ._slice import crop_middle_third
from ._convert import get_fdata

EPS = 1e-8


class Window(NamedTuple):
    vmin: float
    vmax: float


def minmax(img: NiftiLike) -> Window:
    """
    Compute the min-max window for an image.
    """
    data = get_fdata(img)
    vmin, vmax = np.nanmin(data), np.nanmax(data)
    return Window(vmin, vmax)


def center_minmax(img: NiftiLike, idx: int = 0) -> Window:
    """
    Compute the min-max window for just the middle third of the axial slice passing
    through the volume "peak" of mass. If 4d, compute over the `idx` volume.
    """
    data = get_fdata(img)
    check_3d_4d(data)
    if data.ndim == 4:
        data = data[..., idx]
    if data.ndim == 3:
        zidx = np.argmax(np.sum(np.abs(data), axis=(0, 1)))
        data = data[..., zidx]
    data = crop_middle_third(data, axis=(0, 1))
    return minmax(data)
