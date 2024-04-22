"""Tests functionality of multi-view figure generation (not figure content)."""

from pathlib import Path

import nibabel as nib
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
            overlay=nii_3d_img,
        )
        assert isinstance(frame, Image.Image)

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

    def test_4d(self, nii_4d_img: nib.Nifti1Image, tmp_path: Path):
        out_fpath = tmp_path / "test_4d.mp4"
        mv.slice_video(img=nii_4d_img, out=out_fpath, idx=None)
        assert out_fpath.exists()
