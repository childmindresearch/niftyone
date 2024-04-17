from typing import Optional

import nibabel as nib
from PIL import Image

from ..typing import Coord
from ._annotate import annotate as draw_annotation
from ._convert import reorient, scale, topil
from ._slice import slice_volume


def render_slice(
    img: nib.Nifti1Image,
    axis: int,
    coord: Coord,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    height: Optional[int] = 256,
    cmap: str = "gray",
    annotate: bool = True,
    fontsize: int = 14,
) -> Image.Image:
    """Render one slice of a volume as a PIL image."""
    frame = slice_volume(img, coord=coord, axis=axis)
    frame = reorient(frame)
    frame = topil(frame, vmin=vmin, vmax=vmax, cmap=cmap)
    if height:
        frame = scale(frame, height, resample=Image.Resampling.NEAREST)

    # Annotation for coordinate and left side
    if annotate:
        axis_name = "XYZ"[axis]
        label = f"{axis_name}={coord[axis]:.0f}"
        frame = draw_annotation(
            frame, text=label, loc="lower right", size=fontsize, inplace=True
        )
        if axis_name in {"Y", "Z"}:
            frame = draw_annotation(
                frame, text="L", loc="lower left", size=fontsize, inplace=True
            )
    return frame
