import logging

import matplotlib as mpl
import nibabel as nib
import numpy as np
from PIL import Image

from niclips.typing import NiftiLike

EPS = 1e-8


def get_fdata(img: NiftiLike) -> np.ndarray:
    """Get the array data of a nifti-like image."""
    if isinstance(img, nib.nifti1.Nifti1Image):
        img = img.get_fdata()
    return img


def topil(
    data: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "gray",
) -> Image.Image:
    """Convert a numpy array to a PIL image."""
    if isinstance(data, Image.Image):
        return data

    # Assume that 2d arrays need to be normalized and colormapped
    if data.ndim == 2:
        data = normalize(data, vmin=vmin, vmax=vmax)
        data = mpl.colormaps[cmap](data)
        # RGBA -> RGB
        data = data[..., :3]
        data = (255 * data).astype("uint8")

    img = Image.fromarray(data)
    return img


def overlay(
    img1: Image.Image,
    img2: Image.Image,
    alpha: float | None = 0.5,
) -> Image.Image:
    """Overlay two PIL images with alpha compositing."""
    img1 = img1.convert("RGBA")
    img2 = img2.convert("RGBA")

    if alpha:
        img2.putalpha(int(alpha * 255))
    img = Image.alpha_composite(img1, img2)
    return img


def normalize(
    data: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """Normalize data using vmin/vmax if provided, otherwise data min/max."""
    vmin = np.nanmin(data) if vmin is None else vmin
    vmax = np.nanmax(data) if vmax is None else vmax
    scale = vmax - vmin
    if scale > EPS:
        return np.clip((data - vmin) / scale, 0, 1)
    return np.zeros_like(data)


def scale(
    img: Image.Image,
    target_height: int,
    resample: Image.Resampling | None = None,
) -> Image.Image:
    """Scale to targeted height."""
    # Note: Ensure size is even numbered for video codec.
    if not (target_height % 2) == 0:
        target_height += 1
        logging.warning(f"Scaling target height to {target_height}")

    scale = target_height / img.height
    target_width = int(scale * img.width)
    target_width += target_width % 2

    # Resize image
    return img.resize((target_width, target_height), resample=resample)


def to_iso(
    img: Image.Image,
    pixdims: list[float],
    axis: int = 0,
    resample: Image.Resampling | None = None,
) -> Image.Image:
    """Scale frame to isotropic pixels."""
    if axis > 2:
        raise ValueError("Axis must be 0, 1, or 2")

    target_scales = pixdims / np.min(pixdims)
    if axis == 0:
        target_scales = target_scales[1:]
    elif axis == 2:
        target_scales = target_scales[:2]
    else:
        target_scales = target_scales[[0, 2]]
    new_size = (int(img.width * target_scales[0]), int(img.height * target_scales[1]))
    return img.resize(new_size, resample=resample)


def reorient(img: np.ndarray) -> np.ndarray:
    """Reorient image axes from XY (i.e. Nifti-like) to IJ (i.e. typical image-like)."""
    return np.flipud(np.swapaxes(img, 0, 1))


def to_ras(img: nib.nifti1.Nifti1Image) -> nib.nifti1.Nifti1Image:
    """Convert a nifti image to RAS orientation with isotropic resolution."""
    # Grab original filepath
    img_path = img.get_filename()
    # Reorient to RAS
    img = nib.funcs.as_closest_canonical(img)

    # Set filepath to original incase it is needed
    if img_path:
        img.set_filename(img_path)
    return img
