from enum import Enum
from typing import Any, List

import numpy as np


class Align(Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"


def pad_to_size(
    img: np.ndarray,
    size: int,
    axis: int,
    fill_value: Any = 0,
    align: Align = Align.CENTER,
) -> np.ndarray:
    """
    Pad an image to a target size on one of the axes.
    """
    img = np.asarray(img)
    current_size = img.shape[axis]
    assert current_size <= size, f"Size {size} is too small for image"
    if current_size == size:
        return img

    padding = size - current_size
    align = Align(align)
    if align in {Align.LEFT, Align.TOP}:
        padding = (0, padding)
    elif align == {Align.RIGHT, Align.BOTTOM}:
        padding = (padding, 0)
    else:
        padding = (padding // 2, padding - padding // 2)

    padding = [padding if ii == axis else (0, 0) for ii in range(img.ndim)]
    img = np.pad(img, padding, constant_values=fill_value)
    return img


def pad_to_equal(
    imgs: List[np.ndarray],
    axis: int,
    fill_value: Any = 0,
    align: Align = Align.CENTER,
) -> List[np.ndarray]:
    """
    Pad images to be all the same dimension for a given axis.
    """
    imgs = [np.asarray(img) for img in imgs]

    size = max(img.shape[axis] for img in imgs)
    imgs = [
        pad_to_size(img, size, axis=axis, fill_value=fill_value, align=align)
        for img in imgs
    ]
    return imgs


def pad_to_square(img: np.ndarray, fill_value: Any = 0) -> np.ndarray:
    """
    Pad an image to be square.
    """
    img = np.asarray(img)
    h, w = img.shape[:2]
    size = max(w, h)
    axis = 0 if w > h else 1
    img = pad_to_size(img, size=size, axis=axis, fill_value=fill_value)
    return img
