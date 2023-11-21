from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np
from PIL import Image

import niftyone.image as noimg
from niftyone.checks import check_3d, check_3d_4d, check_4d, check_iso_ras
from niftyone.defaults import get_default_coord, get_default_vmin_vmax
from niftyone.io import VideoWriter
from niftyone.typing import Coord, StrPath


def multi_view_frame(
    img: nib.Nifti1Image,
    coords: List[Coord],
    axes: List[int],
    out: Optional[StrPath] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    overlay: Optional[nib.Nifti1Image] = None,
    nrows: int = 1,
    panel_height: Optional[int] = 256,
    cmap: str = "gray",
    overlay_cmap: str = "turbo",
    alpha: float = 0.5,
    fontsize: int = 14,
) -> Image.Image:
    """
    Construct a multi view image panel. Returns a PIL Image.
    """
    check_3d(img)
    check_iso_ras(img)
    if overlay is not None:
        check_3d(overlay)
        overlay = check_iso_ras(overlay)
    vmin, vmax = get_default_vmin_vmax(img, vmin, vmax)

    panels = []
    for coord, axis in zip(coords, axes):
        panel = noimg.render_slice(
            img,
            axis=axis,
            coord=coord,
            vmin=vmin,
            vmax=vmax,
            height=panel_height,
            cmap=cmap,
            annotate=overlay is None,
            fontsize=fontsize,
        )

        if overlay is not None:
            panel_overlay = noimg.render_slice(
                overlay,
                axis=axis,
                coord=coord,
                height=panel_height,
                cmap=overlay_cmap,
                fontsize=fontsize,
            )
            panel = noimg.overlay(panel, panel_overlay, alpha=alpha)

        panels.append(panel)

    # Pad to equal height
    panels = noimg.pad_to_equal(panels, axis=0)
    grid = noimg.image_grid(panels, nrows=nrows)
    grid = noimg.topil(grid)

    if out is not None:
        grid.save(out)
    return grid


def three_view_frame(
    img: nib.Nifti1Image,
    out: Optional[StrPath] = None,
    coord: Optional[Tuple[float, float, float]] = None,
    idx: Optional[int] = 0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    overlay: Optional[nib.Nifti1Image] = None,
    panel_height: Optional[int] = 256,
    cmap: str = "gray",
    overlay_cmap: str = "turbo",
    alpha: float = 0.5,
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
        out=out,
        vmin=vmin,
        vmax=vmax,
        overlay=overlay,
        panel_height=panel_height,
        cmap=cmap,
        overlay_cmap=overlay_cmap,
        alpha=alpha,
        fontsize=fontsize,
    )
    return grid
