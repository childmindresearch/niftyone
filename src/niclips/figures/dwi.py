"""Diffusion MRI figure generation module."""

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.animation import FuncAnimation

import niclips.image as noimg
from niclips.checks import check_4d
from niclips.figures.multi_view import three_view_video
from niclips.typing import StrPath


def _equate_bvals(bvals: np.ndarray, thresh: int) -> np.ndarray:
    """Map bvals within a given threshold to each other."""
    uniq_bvals = sorted(np.unique(bvals).astype(int))
    bval_map = {}

    cur_bval = uniq_bvals[0]
    for bval in uniq_bvals:
        if (bval - cur_bval) > thresh:
            cur_bval = bval
        bval_map[bval] = cur_bval

    return np.array([bval_map[bval] for bval in bvals])


def _get_bval_indices(bvals: np.ndarray, bval: int) -> np.ndarray:
    """Grab indices corresponding to a given bval."""
    idxes = np.argwhere(bvals == bval)
    return idxes[0] if len(idxes) == 1 else idxes


def visualize_qspace(
    dwi: Path,
    out: StrPath | None = None,
    *,
    thresh: int = 10,
    figure: str | None = None,
) -> FuncAnimation:
    """Visualize diffusion gradients in q-space."""
    # Grab paths and check existence
    bvec = dwi.with_suffix("").with_suffix(".bvec")
    bval = dwi.with_suffix("").with_suffix(".bval")
    assert all([bvec.exists(), bval.exists()])

    # Gradient vector
    bvec_data = np.loadtxt(bvec)

    # Equivalent gradient magnitudes
    bval_data = np.loadtxt(bval).astype(int)
    bval_data = _equate_bvals(bval_data, thresh=thresh)

    # Generate animation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot by magnitude
    for idx, val in enumerate(np.unique(bval_data)):
        bval_idxes = _get_bval_indices(bval_data, val)
        ax.scatter(
            bvec_data[0, bval_idxes] * idx,
            bvec_data[1, bval_idxes] * idx,
            bvec_data[2, bval_idxes] * idx,
            alpha=0.5,
        )

    # Settings
    ax.set_title("Diffusion gradients in q-space")
    ax.set_aspect("equal")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Animate
    def _rotate(angle: int) -> None:
        ax.view_init(elev=0, azim=angle, roll=0)

    ani = FuncAnimation(fig, _rotate, frames=np.arange(0, 360, 1), interval=30)

    if out:
        ani.save(out, writer="ffmpeg", fps=30, dpi=150)
    return ani


def three_view_per_shell(
    dwi: Path,
    out: StrPath | None = None,
    *,
    thresh: int = 10,
    replace_str: str = "bval",
) -> list[nib.Nifti1Image]:
    """Generate three-view videos per shell."""
    # Grab bvals
    bval = dwi.with_suffix("").with_suffix(".bval")
    assert bval.exists()
    bval_data = np.loadtxt(bval).astype(int)
    bval_data = _equate_bvals(bval_data, thresh=thresh)

    # Load image
    dwi_data = nib.nifti1.load(dwi)
    dwi_data = noimg.to_iso_ras(dwi_data)

    figs = []
    for val in np.unique(bval_data):
        bval_idxes = _get_bval_indices(bval_data, val)
        dwi_arr = dwi_data.dataobj[:, :, :, bval_idxes]
        # Squeeze if necessary (e.g. more than 1 volume in a shell)
        if len(dwi_arr.shape) == 5:
            if dwi_arr.shape[-1] > 1:
                raise ValueError(f"Diffusion image of the wrong shape {dwi_arr.shape}")
            dwi_arr = np.squeeze(dwi_arr, axis=-1)
        figs.append(nib.Nifti1Image(dwi_arr, affine=dwi_data.affine))
        check_4d(figs[-1])

        if out:
            three_view_video(img=figs[-1], out=str(out).replace(replace_str, f"b{val}"))

    return figs
