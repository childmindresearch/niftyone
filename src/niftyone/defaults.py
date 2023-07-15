from typing import Optional, Tuple

import nibabel as nib

import niftyone.image as noimg


def get_default_coord(img: nib.Nifti1Image) -> Tuple[float, float, float]:
    coord = noimg.peak_of_mass(img, mask=True)
    coord = tuple(coord + [1.0, 0.0, 0.0])
    return coord


def get_default_window(img: nib.Nifti1Image) -> noimg.Window:
    return noimg.center_minmax(img)


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
