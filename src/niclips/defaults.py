"""Get default values."""

import nibabel as nib

import niclips.image as noimg


def get_default_coord(img: nib.Nifti1Image) -> tuple[float, float, float]:
    """Get default coordinates of an image (middle of volume)."""
    coord = noimg.ind2coord(img.affine, img.header["dim"][1:4] // 2)
    coord = tuple(coord)
    return coord


def get_default_window(img: nib.Nifti1Image) -> noimg.Window:
    """Get default window of an image."""
    return noimg.center_minmax(img)


def get_default_vmin_vmax(
    img: nib.Nifti1Image, vmin: float | None = None, vmax: float | None = None
) -> tuple[float, float]:
    """Get default window min/max of an image."""
    if vmin is None or vmax is None:
        window = get_default_window(img)
        if vmin is None:
            vmin = window.vmin
        if vmax is None:
            vmax = window.vmax
    return vmin, vmax
