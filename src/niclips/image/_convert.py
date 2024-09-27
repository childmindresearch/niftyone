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
    img = np.asarray(img)
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
    data = np.asarray(data)
    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)
    data = (data - vmin) / max(vmax - vmin, EPS)
    data = np.clip(data, 0, 1)
    return data


def scale(
    img: Image.Image, height: int, resample: Image.Resampling | None = None
) -> Image.Image:
    """Scale an image to a target height.

    Ensure width is even numbered.
    """
    height += height % 2
    scale = height / img.height
    width = int(scale * img.width) + (int(scale * img.width) % 2)
    size = width, height
    img = img.resize(size, resample=resample)
    return img


def reorient(img: np.ndarray) -> np.ndarray:
    """Reorient image axes from XY (i.e. Nifti-like) to IJ (i.e. typical image-like)."""
    return np.flipud(np.swapaxes(img, 0, 1))


def to_ras(img: nib.nifti1.Nifti1Image) -> nib.nifti1.Nifti1Image:
    """Convert a nifti image to RAS orientation with isotropic resolution."""
    # Grab original filepath
    img_path = img.get_filename()
    # Reorient to RAS
    trgt_ornt = np.array([[0, 1], [1, 1], [2, 1]])
    crnt_ornt = nib.orientations.io_orientation(img.affine)
    xfm = nib.orientations.ornt_transform(crnt_ornt, trgt_ornt)
    reoriented_affine = nib.orientations.inv_ornt_aff(xfm, img.shape)
    new_affine = img.affine @ reoriented_affine
    reoriented_data = nib.orientations.apply_orientation(get_fdata(img), xfm)
    img = nib.Nifti1Image(dataobj=reoriented_data, affine=new_affine)

    # Set filepath to original incase it is needed
    if img_path:
        img.set_filename(img_path)
    return img
