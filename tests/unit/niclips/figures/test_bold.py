"""Tests functionality of figure generation of bold images (not figure content)."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from matplotlib.figure import Figure
from PIL import Image

from niclips.figures import bold as nobold


@pytest.fixture
def nii_bold() -> nib.Nifti1Image:
    img_arr = np.random.rand(10, 10, 10, 3)

    return nib.Nifti1Image(dataobj=img_arr, affine=np.eye(4))


class TestClusterTimeSeries:
    def test_default(self, nii_bold: nib.Nifti1Image):
        clustered_img = nobold.cluster_timeseries(nii_bold)
        assert isinstance(clustered_img, nib.Nifti1Image)


class TestCarpetPlot:
    def test_default(self, nii_bold: nib.Nifti1Image):
        carpet_plot = nobold.carpet_plot(nii_bold)

        assert isinstance(carpet_plot, Figure)

    def test_label(self, nii_bold: nib.Nifti1Image):
        label_img = nib.Nifti1Image(
            dataobj=np.random.rand(10, 10, 10), affine=np.eye(4)
        )
        carpet_plot = nobold.carpet_plot(nii_bold, label=label_img)

        assert isinstance(carpet_plot, Figure)

    def test_save(self, nii_bold: nib.Nifti1Image, tmp_path: Path):
        out_fpath = tmp_path / "test_carpet_plot.png"
        nobold.carpet_plot(nii_bold, out=out_fpath)
        assert out_fpath.exists()


class TestBoldMeanStd:
    def test_default(self, nii_bold: nib.Nifti1Image):
        grid = nobold.bold_mean_std(nii_bold)

        assert isinstance(grid, Image.Image)

    def test_save(self, nii_bold: nib.Nifti1Image, tmp_path: Path):
        out_fpath = tmp_path / "test_bold_mean_std.png"
        nobold.bold_mean_std(nii_bold, out=out_fpath)

        assert out_fpath.exists()
