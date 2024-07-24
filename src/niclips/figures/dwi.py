"""Diffusion MRI figure generation module."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from niclips.typing import StrPath


def _equate_bvals(bvals: list[int], thresh: int) -> np.ndarray:
    """Map bvals within a given threshold to each other."""
    uniq_bvals = sorted(np.unique(bvals))
    bval_map = {}

    cur_bval = uniq_bvals[0]
    idx = 0
    for bval in uniq_bvals:
        if (bval - cur_bval) > thresh:
            cur_bval = bval
            idx += 1
        bval_map[bval] = idx

    return np.array([bval_map[bval] for bval in bvals])


def _get_bval_indices(bvals: np.ndarray, bval: int) -> np.ndarray:
    """Grab indices corresponding to a given bval."""
    return np.argwhere(bvals == bval)


def visualize_shells(
    dwi: Path,
    out: StrPath | None = None,
    *,
    thresh: int = 10,
) -> None:
    """Visualize diffusion gradients in q-space."""
    # Grab paths and check existence
    bvec = dwi.with_suffix("").with_suffix(".bvec")
    bval = dwi.with_suffix("").with_suffix(".bval")
    assert all([bvec.exists(), bval.exists()])

    # Gradient vector
    bvec_data = np.loadtxt(bvec)

    # Equivalent gradient magnitudes
    bval_data = np.loadtxt(bval, dtype=int)
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
        ani.save(out, writer="ffmpeg", fps=30)
