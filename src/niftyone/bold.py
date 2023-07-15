import math
from typing import Optional

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.cluster import KMeans

import niftyone.image as noimg
from niftyone.checks import check_3d, check_4d, check_iso_ras
from niftyone.defaults import get_default_coord, get_default_vmin_vmax
from niftyone.typing import StrPath

EPS = 1e-6
plt.style.use("bmh")


def cluster_timeseries(
    bold: nib.Nifti1Image,
    n_clusters: int = 3,
    n_samples: int = 10000,
    seed: int = 42,
) -> nib.Nifti1Image:
    """
    Segment a BOLD volume by k-means clustering to the timeseries.
    """
    rng = np.random.default_rng(seed)

    # bold data and mean volume
    bold_data = bold.get_fdata()
    bold_mean = bold_data.mean(axis=-1)

    # rough mask
    mask = bold_mean > bold_mean.mean()
    mask_data = bold_data[mask]

    # fit clustering to subset of timeseries
    clust = KMeans(n_clusters=n_clusters, n_init="auto", random_state=seed)
    indices = rng.permutation(len(mask_data))[:n_samples]
    clust.fit(mask_data[indices])

    # predict cluster for full mask data
    mask_label = clust.predict(mask_data)

    # sort labels by t-SNR
    mask_tsnr = bold_mean[mask] / (mask_data.std(axis=-1) + EPS)
    class_tsnr = [np.mean(mask_tsnr[mask_label == idx]) for idx in range(n_clusters)]
    ranking = np.argsort(np.argsort(class_tsnr))
    mask_label = ranking[mask_label]

    # construct label volume
    label = np.full(bold.shape[:3], np.nan)
    label[mask] = mask_label
    label = nib.Nifti1Image(label, affine=bold.affine)
    return label


def carpet_plot(
    bold: nib.Nifti1Image,
    out: Optional[StrPath] = None,
    label: Optional[nib.Nifti1Image] = None,
    n_voxels: int = 2000,
    seed: int = 42,
    label_cmap: str = "brg",
    alpha: float = 0.3,
):
    """
    BOLD "carpet" plot showing timeseries for a subset of voxels.
    """
    rng = np.random.default_rng(seed)

    check_4d(bold)
    check_iso_ras(bold)
    if label is None:
        label = cluster_timeseries(bold)
    check_3d(label)
    check_iso_ras(label)

    # get bold data and mean image
    bold_data = bold.get_fdata()
    bold_mean_data = bold_data.mean(axis=-1)
    bold_mean = nib.Nifti1Image(bold_mean_data, affine=bold.affine)

    # axial bold mean and label overlay
    coord = get_default_coord(bold_mean)
    vmin, vmax = get_default_vmin_vmax(bold_mean)
    panel_mean = noimg.render_slice(
        bold_mean,
        axis=2,
        coord=coord,
        vmin=vmin,
        vmax=vmax,
        fontsize=18,
    )
    panel_label = noimg.render_slice(
        label,
        axis=2,
        coord=coord,
        cmap=label_cmap,
    )
    panel_label = noimg.overlay(panel_mean, panel_label, alpha=alpha)

    # assume nan is background; apply mask
    label_data = label.get_fdata()
    mask = ~np.isnan(label_data)
    mask_data = bold_data[mask]
    mask_label = label_data[mask]

    # random sample of voxels
    indices = rng.permutation(len(mask_data))[:n_voxels]
    example_data = mask_data[indices]
    example_label = mask_label[indices]

    # order by cluster
    order = np.argsort(example_label)
    example_data = example_data[order]
    example_label = example_label[order]

    # centered and scaled carpet
    carpet = example_data - example_data.mean(axis=1, keepdims=True)
    carpet = carpet / (carpet.std() + EPS)

    # generate plots
    fig = plt.figure(layout="tight")
    gs = GridSpec(1, 4, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(panel_mean)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(panel_label)
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 2:])
    h, w = carpet.shape
    # extra space for label
    label_w = math.ceil(0.05 * w)
    ax3.imshow(example_label[:, None], cmap=label_cmap, extent=(-label_w, 0, 0, h))
    ax3.imshow(carpet, cmap="gray", vmin=-2.0, vmax=2.0, extent=(0, w, 0, h))
    ax3.set_xlim(-label_w, w)
    ax3.set_ylim(0, h)
    ax3.set_yticks([])
    # HACK: annoying hack to make the aspect ratio work
    aspect = (label_w + w) / h / 2
    ax3.set_aspect(aspect)
    ax3.tick_params(colors="w", labelsize=8, direction="in", pad=-8)

    fig.set_facecolor("gray")

    if out is not None:
        fig.savefig(out, bbox_inches="tight", dpi=150)
    return fig
