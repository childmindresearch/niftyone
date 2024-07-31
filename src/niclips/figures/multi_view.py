"""Generation of different multi-views."""

from collections.abc import Sequence
from pathlib import Path

import nibabel as nib
import numpy as np
from bids2table import BIDSEntities
from PIL import Image

import niclips.image as noimg
from niclips.checks import check_3d, check_3d_4d, check_4d, check_iso_ras
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
    overlay: nib.Nifti1Image | None = None,
    nrows: int = 1,
    panel_height: int | None = 256,
    cmap: str = "gray",
    overlay_cmap: str = "turbo",
    alpha: float = 0.5,
    fontsize: int = 14,
    figure: str | None = None,
) -> Image.Image:
    """Construct a multi view image panel. Returns a PIL Image."""
    check_3d(img)
    check_iso_ras(img)
    if overlay:
        check_3d(overlay)
        check_iso_ras(overlay)
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

        if overlay:
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
    panels_list = noimg.pad_to_equal(panels, axis=0)
    grid = noimg.image_grid(panels_list, nrows=nrows)
    grid_img = noimg.topil(grid)

    if out:
        grid_img.save(out)
    return grid_img


def three_view_overlay_frame(
    img: Path,
    out: StrPath | None = None,
    *,
    entities: dict[str, str] = {"desc": "brain", "suffix": "mask"},
    figure: str | None = None,
) -> Image.Image:
    """Construct overlay with image.

    Default is to overlay with mask with similar entities
    (e.g. entities = {"desc": "brain", "suffix": "mask"}).

    For example:
    'sub-01/anat/sub-01_ses-01_run-1_T1w.nii.gz' will be overlaid with
    'sub-01/anat/sub-01_ses-01_run-1_desc-brain_mask.nii.gz'
    """
    # Grab mask based on provided entities
    img_entities = BIDSEntities.from_path(img)
    img_base_path = Path(str(img).replace(str(img_entities.to_path()), ""))
    overlay_entities = img_entities.with_update(entities)
    overlay = img_base_path.joinpath(overlay_entities.to_path())
    assert overlay.exists()

    # Load image
    img_data = nib.nifti1.load(img)
    img_data = noimg.to_iso_ras(img_data)
    overlay_data = nib.nifti1.load(overlay)
    overlay_data = noimg.to_iso_ras(overlay_data)

    grid = three_view_frame(
        img=img_data,
        out=out,
        overlay=overlay_data,
    )

    return grid


def three_view_frame(
    img: NiftiLike,
    out: StrPath | None = None,
    *,
    coord: tuple[float, float, float] | None = None,
    idx: int | None = 0,
    vmin: float | None = None,
    vmax: float | None = None,
    overlay: nib.Nifti1Image | None = None,
    panel_height: int | None = 256,
    cmap: str = "gray",
    overlay_cmap: str = "turbo",
    alpha: float = 0.5,
    fontsize: int = 14,
    figure: str | None = None,
) -> Image.Image:
    """Construct a three view image panel. Returns a PIL Image."""
    check_3d_4d(img)
    if img.ndim == 4:
        img = noimg.index_img(img, idx=idx)
    assert isinstance(img, nib.Nifti1Image)

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
    *,
    coord: tuple[float, float, float] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    panel_height: int | None = 256,
    cmap: str = "gray",
    fontsize: int = 14,
    figure: str | None = None,
) -> None:
    """Save a three view panel video."""
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
    img: NiftiLike,
    out: StrPath,
    *,
    axis: int = 2,
    idx: int | None = 0,
    vmin: float | None = None,
    vmax: float | None = None,
    panel_height: int | None = 256,
    cmap: str = "gray",
    fontsize: int = 14,
    figure: str | None = None,
) -> None:
    """Save video scrolling through range of slices."""
    check_3d_4d(img)
    if img.ndim == 4:
        img = noimg.index_img(img, idx=idx)
    assert isinstance(img, nib.Nifti1Image)
    check_iso_ras(img)
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
            writer.put(frame)
