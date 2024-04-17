from functools import lru_cache
from importlib import resources
from typing import Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def annotate(
    img: Union[np.ndarray, Image.Image],
    text: str,
    loc: str,
    size: int = 10,
    fill: str = "white",
    inplace: bool = False,
) -> Image.Image:
    """Annotate an image with text.

    Locations follow the convention from pyplot, e.g. "upper left", "lower right".
    However, only a subset of locations are currently implemented.
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    assert isinstance(img, Image.Image)

    offset = 2
    if loc == "upper left":
        xy = offset, offset
        anchor = "lt"
    elif loc == "upper right":
        xy = img.width - offset, offset
        anchor = "rt"
    elif loc == "lower left":
        xy = offset, img.height - offset
        anchor = "lb"
    elif loc == "lower right":
        xy = img.width - offset, img.height - offset
        anchor = "rb"
    else:
        raise ValueError(f"Unsupported loc {loc}")

    if not inplace:
        img = img.copy()
    draw = ImageDraw.Draw(img)
    font = _get_font(size=size)
    draw.text(xy, text, fill=fill, font=font, anchor=anchor)
    return img


@lru_cache
def _get_font(size: int) -> ImageFont.FreeTypeFont:
    with resources.path("niclips.image._resources", "SpaceMono-Regular.ttf") as p:
        font = ImageFont.truetype(str(p), size=size)
    return font
