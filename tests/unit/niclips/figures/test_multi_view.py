"""Tests functionality of multi-view figure generation (not figure content)."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from _pytest.logging import LogCaptureFixture
from PIL import Image

from niclips.figures import multi_view as mv


class TestMultiViewFrame:
    def test_default(self, nii_3d_img: nib.Nifti1Image):
        frame = mv.multi_view_frame(img=nii_3d_img, coords=[(0, 0, 0)], axes=[0])

        assert isinstance(frame, Image.Image)

    def test_overlay(self, nii_3d_img: nib.Nifti1Image):
        frame = mv.multi_view_frame(
            img=nii_3d_img,
            coords=[(0, 0, 0)],
            axes=[0],
            overlay=[nii_3d_img, nii_3d_img],
        )
        assert isinstance(frame, Image.Image)

    def test_overlay_cmap(self, nii_3d_img: nib.Nifti1Image):
        frame = mv.multi_view_frame(
            img=nii_3d_img,
            coords=[(0, 0, 0)],
            axes=[0],
            overlay=[nii_3d_img, nii_3d_img],
            overlay_cmap=["brg", "turbo"],
        )
        assert isinstance(frame, Image.Image)

    def test_overlay_cmap_warning(
        self, nii_3d_img: nib.Nifti1Image, caplog: LogCaptureFixture
    ):
        mv.multi_view_frame(
            img=nii_3d_img,
            coords=[(0, 0, 0)],
            axes=[0],
            overlay=[nii_3d_img, nii_3d_img],
            overlay_cmap=["brg"],
        )
        assert "More overlays" in caplog.text

    def test_save_frame(self, nii_3d_img: nib.Nifti1Image, tmp_path: Path):
        out_path = tmp_path / "test.png"
        mv.multi_view_frame(img=nii_3d_img, coords=[(0, 0, 0)], axes=[0], out=out_path)

        assert out_path.exists()


class TestThreeViewFrame:
    def test_3d(self, nii_3d_img: nib.Nifti1Image):
        grid = mv.three_view_frame(img=nii_3d_img)
        assert isinstance(grid, Image.Image)

    def test_4d(self, nii_4d_img: nib.Nifti1Image):
        grid = mv.three_view_frame(img=nii_4d_img)
        assert isinstance(grid, Image.Image)

    def test_overlay(self, nii_4d_img: nib.Nifti1Image):
        grid = mv.three_view_frame(nii_4d_img, overlay=[nii_4d_img, nii_4d_img])
        assert isinstance(grid, Image.Image)


class TestThreeViewVideo:
    def test_default(self, nii_4d_img: nib.Nifti1Image, tmp_path: Path):
        out_fpath = tmp_path / "test_three_view.mp4"
        mv.three_view_video(nii_4d_img, out=out_fpath)
        assert out_fpath.exists()


class TestSliceVideo:
    def test_3d(self, nii_3d_img: nib.Nifti1Image, tmp_path: Path):
        out_fpath = tmp_path / "test_3d.mp4"
        mv.slice_video(img=nii_3d_img, out=out_fpath)
        assert out_fpath.exists()

    @pytest.mark.parametrize("idx", [(None), (0), (1), (2)])
    def test_4d(self, tmp_path: Path, idx: int | None):
        out_fpath = tmp_path / "test_4d.mp4"
        test_img = nib.Nifti1Image(np.random.rand(10, 10, 10, 3), affine=np.eye(4))
        mv.slice_video(img=test_img, out=out_fpath, idx=idx)
        assert out_fpath.exists()

    def test_overlay(self, tmp_path: Path):
        out_fpath = tmp_path / "test_overlay.mp4"
        test_img = nib.Nifti1Image(np.random.rand(10, 10, 10, 3), affine=np.eye(4))
        mv.slice_video(img=test_img, out=out_fpath, overlay=[test_img])
        assert out_fpath.exists()
