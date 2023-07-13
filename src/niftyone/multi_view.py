from typing import List, Optional, Tuple

import nibabel as nib
from PIL import Image

import niftyone.image as noimg
from niftyone.checks import check_3d, check_3d_4d, check_4d
from niftyone.io import VideoWriter
from niftyone.typing import StrPath

Coord = Tuple[float, float, float]


def multi_view_frame(
    img: nib.Nifti1Image,
    coords: List[Coord],
    axes: List[int],
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    nrows: int = 1,
    panel_height: Optional[int] = 256,
    cmap: str = "gray",
    fontsize: int = 14,
) -> Image.Image:
    """
    Construct a multi view image panel. Returns a PIL Image.
    """
    check_3d(img)
    img = noimg.to_iso_ras(img)

    if vmin is None or vmax is None:
        window = get_default_window(img)
        if vmin is None:
            vmin = window.vmin
        if vmax is None:
            vmax = window.vmax

    panels = []
    for coord, axis in zip(coords, axes):
        panel = noimg.slice_volume(img, coord=coord, axis=axis)
        panel = noimg.reorient(panel)
        panels.append(panel)

    # Pad to equal height
    panels = noimg.pad_to_equal(panels, axis=0)

    rendered = []
    for panel, coord, axis in zip(panels, coords, axes):
        img = noimg.topil(panel, vmin=vmin, vmax=vmax, cmap=cmap)
        if panel_height:
            img = noimg.scale(img, panel_height, resample=Image.Resampling.NEAREST)

        # Annotation for coordinate and left side
        axis_name = "XYZ"[axis]
        label = f"{axis_name}={coord[axis]:.0f}"
        img = noimg.annotate(
            img, text=label, loc="lower right", size=fontsize, inplace=True
        )
        if axis_name in {"Y", "Z"}:
            img = noimg.annotate(
                img, text="L", loc="lower left", size=fontsize, inplace=True
            )
        rendered.append(img)

    grid = noimg.image_grid(rendered, nrows=nrows)
    grid = noimg.topil(grid)
    return grid


def three_view_frame(
    img: nib.Nifti1Image,
    coord: Optional[Tuple[float, float, float]] = None,
    idx: Optional[int] = 0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    panel_height: Optional[int] = 256,
    cmap: str = "gray",
    fontsize: int = 14,
) -> Image.Image:
    """
    Construct a three view image panel. Returns a PIL Image.
    """
    check_3d_4d(img)
    if img.ndim == 4:
        img = noimg.index_img(img, idx=idx)

    if coord is None:
        coord = get_default_coord(img)

    grid = multi_view_frame(
        img,
        coords=3 * [coord],
        # ax, cor, sag
        axes=[2, 1, 0],
        vmin=vmin,
        vmax=vmax,
        panel_height=panel_height,
        cmap=cmap,
        fontsize=fontsize,
    )
    return grid


def three_view_video(
    img: nib.Nifti1Image,
    out: StrPath,
    coord: Optional[Tuple[float, float, float]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    panel_height: Optional[int] = 256,
    cmap: str = "gray",
    fontsize: int = 14,
):
    """
    Save a three view panel video.
    """
    check_4d(img)

    img_mid = noimg.index_img(img, idx=None)
    if coord is None:
        coord = get_default_coord(img_mid)

    if vmin is None or vmax is None:
        window = get_default_window(img_mid)
        if vmin is None:
            vmin = window.vmin
        if vmax is None:
            vmax = window.vmax

    with VideoWriter(out, fps=10) as writer:
        for idx in range(img.shape[-1]):
            frame = three_view_frame(
                img,
                coord=coord,
                idx=idx,
                vmin=vmin,
                vmax=vmax,
                panel_height=panel_height,
                cmap=cmap,
                fontsize=fontsize,
            )
            frame = noimg.annotate(
                frame, text=f"T={idx}", loc="upper right", size=fontsize
            )
            writer.put(frame)


def get_default_coord(img: nib.nifti1.Nifti1Image) -> Tuple[float, float, float]:
    coord = noimg.peak_of_mass(img, mask=True)
    coord = tuple(coord + [1.0, 0.0, 0.0])
    return coord


def get_default_window(img: nib.nifti1.Nifti1Image) -> noimg.Window:
    return noimg.center_minmax(img)
