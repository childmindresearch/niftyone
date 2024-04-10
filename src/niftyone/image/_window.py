"""Utilities for normalizing value ranges."""

from typing import NamedTuple

import numpy as np

from niftyone.checks import check_3d_4d
from niftyone.typing import NiftiLike

from ._centroid import peak_of_mass
from ._convert import get_fdata
from ._slice import crop_middle_third, index_img

EPS = 1e-8


class Window(NamedTuple):
    vmin: float
    vmax: float


def minmax(img: NiftiLike) -> Window:
    """Compute the min-max window for an image."""
    data = get_fdata(img)
    vmin, vmax = np.nanmin(data), np.nanmax(data)
    return Window(vmin, vmax)


def center_minmax(img: NiftiLike) -> Window:
    """Compute min-max window over center of axial slice; compute over `idx` if 4D."""
    check_3d_4d(img)
    if img.ndim == 4:
        img = index_img(img, idx=None)
    data = get_fdata(img)

    centroid = peak_of_mass(data, mask=True)
    data = data[..., centroid[2]]
    data = crop_middle_third(data, axis=(0, 1))
    return minmax(data)
