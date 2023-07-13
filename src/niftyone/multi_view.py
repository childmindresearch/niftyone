from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np
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
    vmin, vmax = get_default_vmin_vmax(img, vmin, vmax)

    panels = []
    for coord, axis in zip(coords, axes):
        panel = noimg.slice_volume(img, coord=coord, axis=axis)
        panel = noimg.reorient(panel)
        panels.append(panel)

    # Pad to equal height
    panels = noimg.pad_to_equal(panels, axis=0)

    rendered = []
    for panel, coord, axis in zip(panels, coords, axes):
        panel = render_frame(
            panel,
            axis=axis,
            coord=coord,
            vmin=vmin,
            vmax=vmax,
            height=panel_height,
            cmap=cmap,
            fontsize=fontsize,
        )
        rendered.append(panel)

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

    if coord is None:
        coord = get_default_coord(img)
    vmin, vmax = get_default_vmin_vmax(img, vmin, vmax)

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


def slice_video(
    img: nib.Nifti1Image,
    out: StrPath,
    axis: int = 2,
    idx: Optional[int] = 0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    panel_height: Optional[int] = 256,
    cmap: str = "gray",
    fontsize: int = 14,
):
    check_3d_4d(img)
    if img.ndim == 4:
        img = noimg.index_img(img, idx=idx)
    img = noimg.to_iso_ras(img)
    vmin, vmax = get_default_vmin_vmax(img, vmin, vmax)

    # Find range of slices that intersect with a rough mask
    data = noimg.get_fdata(img)
    mask = data > data.mean()
    other_axes = tuple([ii for ii in range(3) if ii != axis])
    indices = np.any(mask, axis=other_axes).nonzero()[0]
    start, stop = indices[0], indices[-1]

    # Initial coord
    coord = [0.0, 0.0, 0.0]
    ind = noimg.coord2ind(img.affine, coord)

    with VideoWriter(out, fps=10) as writer:
        for idx in range(start, stop + 1):
            ind[axis] = idx
            coord = noimg.ind2coord(img.affine, ind)

            frame = noimg.slice_volume(img, coord=coord, axis=axis)
            frame = noimg.reorient(frame)
            frame = render_frame(
                frame,
                axis=axis,
                coord=coord,
                vmin=vmin,
                vmax=vmax,
                height=panel_height,
                cmap=cmap,
                fontsize=fontsize,
            )
            writer.put(frame)


def render_frame(
    img: np.ndarray,
    axis: int,
    coord: Coord,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    height: Optional[int] = 256,
    cmap: str = "gray",
    fontsize: int = 14,
) -> Image.Image:
    """
    Render one frame as a PIL image.
    """
    img = noimg.topil(img, vmin=vmin, vmax=vmax, cmap=cmap)
    if height:
        img = noimg.scale(img, height, resample=Image.Resampling.NEAREST)

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
    return img


def get_default_vmin_vmax(
    img: nib.Nifti1Image, vmin: Optional[float] = None, vmax: Optional[float] = None
) -> Tuple[float, float]:
    if vmin is None or vmax is None:
        window = get_default_window(img)
        if vmin is None:
            vmin = window.vmin
        if vmax is None:
            vmax = window.vmax
    return vmin, vmax


def get_default_coord(img: nib.Nifti1Image) -> Tuple[float, float, float]:
    coord = noimg.peak_of_mass(img, mask=True)
    coord = tuple(coord + [1.0, 0.0, 0.0])
    return coord


def get_default_window(img: nib.Nifti1Image) -> noimg.Window:
    return noimg.center_minmax(img)
