import math
from collections.abc import Sequence
from typing import List

import numpy as np
from PIL import Image

from ._pad import Align, pad_to_equal


def image_grid(
    imgs: List[np.ndarray],
    nrows: int = 1,
    pad: int = 2,
    fill_value: int = 0,
) -> np.ndarray:
    """Combine a list of images, possibly different sizes, into a grid."""
    rows = []
    ncols = math.ceil(len(imgs) / nrows)
    for ii in range(nrows):
        row_imgs = imgs[ii * ncols : (ii + 1) * ncols]
        row = stack_images(row_imgs, axis=1, pad=pad, fill_value=fill_value)
        rows.append(row)

    if nrows == 1:
        grid = rows[0]
    else:
        grid = stack_images(
            rows, axis=0, pad=0, fill_value=fill_value, align=Align.LEFT
        )
    return grid


def stack_images(
    imgs: Sequence[np.ndarray | Image.Image],
    axis: int = 1,
    pad: int = 2,
    fill_value: int = 0,
    align: Align = Align.CENTER,
) -> np.ndarray:
    """Stack a list of images along an axis."""
    imgs = [np.asarray(img) for img in imgs]

    ndims = {img.ndim for img in imgs}
    assert len(ndims) == 1, "Images have different ndims"
    ndim = ndims.pop()
    assert ndim in {2, 3}, "Expected images to have 2 or 3 dims"
    assert axis in {0, 1}, f"Invalid axis {axis}"

    # Pad images to equal size on non-concatenation axis
    other_axis = (axis + 1) % 2
    imgs = pad_to_equal(imgs, axis=other_axis, fill_value=fill_value, align=align)

    # Then pad on all sides
    if pad:
        padding = [(pad, pad) if ii < 2 else (0, 0) for ii in range(ndim)]
        imgs = [np.pad(img, padding) for img in imgs]

    stacked = np.concatenate(imgs, axis=axis)
    return stacked
