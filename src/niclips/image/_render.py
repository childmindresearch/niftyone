import nibabel as nib
from PIL import Image

from ..typing import Coord
from ._annotate import annotate as draw_annotation
from ._convert import reorient, scale, to_iso, topil
from ._slice import slice_volume


def render_slice(
    img: nib.Nifti1Image,
    axis: int,
    coord: Coord,
    vmin: float | None = None,
    vmax: float | None = None,
    height: int | None = 256,
    resample: Image.Resampling | None = None,
    cmap: str = "gray",
    annotate: bool = True,
    fontsize: int = 14,
) -> Image.Image:
    """Render one slice of a volume as a PIL image."""
    frame = slice_volume(img, coord=coord, axis=axis)
    frame = reorient(frame)
    frame = topil(frame, vmin=vmin, vmax=vmax, cmap=cmap)
    frame = to_iso(
        frame, pixdims=img.header["pixdim"][1:4], axis=axis, resample=resample
    )
    if height:
        # Get pixel dimensions of slice
        frame = scale(frame, target_height=height, resample=resample)

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
