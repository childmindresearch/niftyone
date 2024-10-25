"""Generation of different multi-views."""

import logging
from collections.abc import Sequence
from itertools import zip_longest

import nibabel as nib
import numpy as np
from PIL import Image

import niclips.image as noimg
from niclips.checks import check_3d, check_3d_4d, check_4d, check_ras
from niclips.defaults import get_default_coord, get_default_vmin_vmax
from niclips.io import VideoWriter
from niclips.typing import Coord, NiftiLike, StrPath


def multi_view_frame(
    img: nib.Nifti1Image,
    out: StrPath | None = None,
    *,
    coords: Sequence[Coord],
    axes: list[int],
    vmin: float | None = None,
    vmax: float | None = None,
    overlay: nib.Nifti1Image | list[nib.Nifti1Image] | None = None,
    nrows: int = 1,
    panel_height: int | None = 256,
    cmap: str = "gray",
    overlay_cmap: str | list[str] = ["turbo"],
    alpha: float = 0.5,
    fontsize: int = 14,
    **kwargs,
) -> Image.Image:
    """Construct a multi view image panel. Returns a PIL Image."""
    check_3d(img)
    check_ras(img)

    overlay = [overlay] if isinstance(overlay, nib.Nifti1Image) else (overlay or [])
    overlay_cmap = [overlay_cmap] if isinstance(overlay_cmap, str) else overlay_cmap
    if len(overlay) > 0:
        for ov in overlay:
            check_3d(ov)
            check_ras(ov)

    vmin, vmax = get_default_vmin_vmax(img, vmin, vmax)
    panels: list[Image.Image] = []
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

        if len(overlay) > 0:
            if len(overlay_cmap) < len(overlay):
                logging.warning(
                    "More overlays than overlay color maps- will use 'turbo'",
                )
            for ov, ov_cmap in zip_longest(overlay, overlay_cmap):
                panel_overlay = noimg.render_slice(
                    ov,
                    axis=axis,
                    coord=coord,
                    height=panel_height,
                    cmap=ov_cmap or "turbo",
                    annotate=False,
                    fontsize=fontsize,
                )
                panel = noimg.overlay(panel, panel_overlay, alpha)

        panels.append(panel)

    # Pad to equal height
    panels_list = noimg.pad_to_equal(panels, axis=0)
    grid = noimg.image_grid(panels_list, nrows=nrows)
    grid_img = noimg.topil(grid)

    if out:
        grid_img.save(out)
    return grid_img


def three_view_frame(
    img: NiftiLike,
    out: StrPath | None = None,
    coord: Coord | None = None,
    idx: int | None = 0,
    vmin: float | None = None,
    vmax: float | None = None,
    overlay: nib.Nifti1Image | list[nib.Nifti1Image] | None = None,
    panel_height: int | None = 256,
    cmap: str = "gray",
    overlay_cmap: str | list[str] = ["turbo"],
    alpha: float = 0.5,
    fontsize: int = 14,
    **kwargs,
) -> Image.Image:
    """Construct a three view image panel. Returns a PIL Image."""
    check_3d_4d(img)
    if img.ndim == 4:
        img = noimg.index_img(img, idx=idx)
    assert isinstance(img, nib.Nifti1Image)

    overlay = [overlay] if isinstance(overlay, nib.Nifti1Image) else (overlay or [])
    overlay_cmap = [overlay_cmap] if isinstance(overlay_cmap, str) else overlay_cmap
    if len(overlay) > 0:
        for ov_idx, ov in enumerate(overlay):
            if ov.ndim == 4:
                overlay[ov_idx] = noimg.index_img(ov, idx=idx)

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


def three_view_video(
    img: nib.Nifti1Image,
    out: StrPath,
    coord: tuple[float, float, float] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    overlay: nib.Nifti1Image | list[nib.Nifti1Image] | None = None,
    panel_height: int | None = 256,
    cmap: str = "gray",
    overlay_cmap: str | list[str] = ["turbo"],
    fontsize: int = 14,
    **kwargs,
) -> None:
    """Save a three view panel video."""
    check_4d(img)

    overlay = [overlay] if isinstance(overlay, nib.Nifti1Image) else (overlay or [])
    overlay_cmap = [overlay_cmap] if isinstance(overlay_cmap, str) else overlay_cmap

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
                overlay=overlay,
                panel_height=panel_height,
                cmap=cmap,
                overlay_cmap=overlay_cmap,
                fontsize=fontsize,
            )
            frame = noimg.annotate(
                frame, text=f"T={idx}", loc="upper right", size=fontsize
            )
            writer.put(frame)


def slice_video(
    img: NiftiLike,
    out: StrPath,
    axis: int = 2,
    idx: int | None = 0,
    vmin: float | None = None,
    vmax: float | None = None,
    overlay: nib.Nifti1Image | list[nib.Nifti1Image] | None = None,
    panel_height: int | None = 256,
    cmap: str = "gray",
    overlay_cmap: list[str] = ["brg"],
    fontsize: int = 14,
    alpha: float = 0.3,
    **kwargs,
) -> None:
    """Save video scrolling through range of slices."""
    check_3d_4d(img)
    if img.ndim == 4:
        img = noimg.index_img(img, idx=idx)
    assert isinstance(img, nib.Nifti1Image)
    check_ras(img)

    overlay = [overlay] if isinstance(overlay, nib.Nifti1Image) else (overlay or [])
    if len(overlay) > 0:
        for ov_idx, ov in enumerate(overlay):
            if ov.ndim == 4:
                ov = noimg.index_img(ov, idx=idx)
                check_ras(ov)
                overlay[ov_idx] = ov

    vmin, vmax = get_default_vmin_vmax(img, vmin, vmax)

    # Find range of slices that intersect with a rough mask
    data = noimg.get_fdata(img)
    mask = data > data.mean()
    other_axes = tuple([ii for ii in range(3) if ii != axis])
    indices = np.any(mask, axis=other_axes).nonzero()[0]
    start, stop = indices[0], indices[-1]

    # Initial coord
    coord: Coord = (0.0, 0.0, 0.0)
    ind = noimg.coord2ind(img.affine, coord)

    with VideoWriter(out, fps=10) as writer:
        for idx in range(start, stop + 1):
            ind[axis] = idx
            coord = noimg.ind2coord(img.affine, ind)

            frame = noimg.render_slice(
                img,
                axis=axis,
                coord=coord,
                vmin=vmin,
                vmax=vmax,
                height=panel_height,
                cmap=cmap,
                fontsize=fontsize,
            )

            if len(overlay) > 0:
                if len(overlay_cmap) < len(overlay):
                    logging.warning(
                        "More overlays than overlay color maps- will use 'brg'",
                    )
                for ov, ov_cmap in zip_longest(overlay, overlay_cmap):
                    frame_overlay = noimg.render_slice(
                        ov,
                        axis=axis,
                        coord=coord,
                        cmap=ov_cmap or "brg",
                        fontsize=fontsize,
                    )
                    frame = noimg.overlay(frame, frame_overlay, alpha=alpha)

            writer.put(frame)
