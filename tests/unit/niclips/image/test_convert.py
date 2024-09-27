from copy import deepcopy

import nibabel as nib
import numpy as np
import pytest
from PIL import Image

import niclips.image._convert as noconvert


class TestGetData:
    def test_nifti(self, nii_3d_img: nib.Nifti1Image):
        img = noconvert.get_fdata(nii_3d_img)

        assert isinstance(img, np.ndarray)

    def test_array(self, img_array: np.ndarray):
        img = noconvert.get_fdata(img_array)

        assert isinstance(img, np.ndarray)


class TestTopil:
    def test_pil_image(self, img_pil: Image.Image):
        img = noconvert.topil(img_pil)

        assert img is img_pil

    def test_array(self, img_array: np.ndarray):
        img = noconvert.topil(img_array)

        assert isinstance(img, Image.Image)

    def test_2d_array(self, img_array: np.ndarray):
        img = noconvert.topil(img_array[:, :, :2])

        assert isinstance(img, Image.Image)


class TestOverlay:
    def test_no_alpha(self, img_pil: Image.Image):
        img1 = img_pil
        img2 = deepcopy(img_pil)

        new_img = noconvert.overlay(img1, img2, alpha=None)

        assert new_img.mode == "RGBA"
        assert isinstance(new_img, Image.Image)

    def test_alpha(self, img_pil: Image.Image):
        img1 = img_pil
        img2 = deepcopy(img_pil)

        new_img = noconvert.overlay(img1, img2, alpha=0.5)

        assert new_img.mode == "RGBA"
        assert isinstance(new_img, Image.Image)


class TestNormalize:
    def test_data_only(self, img_array: np.ndarray):
        img_array[0, 0, 0] = 10
        data = noconvert.normalize(img_array)

        assert isinstance(data, np.ndarray)
        assert 0.0 <= np.all(data) <= 1.0

    def test_data_vmin_vmax(self, img_array: np.ndarray):
        img_array[0, 0, 0] = 10
        data = noconvert.normalize(img_array, vmin=0.2, vmax=9.8)

        assert isinstance(data, np.ndarray)
        assert 0.0 <= np.all(data) <= 1.0


class TestScale:
    def test_no_resample(self, img_pil: Image.Image):
        img = noconvert.scale(img=img_pil, height=120)

        expected_scale = 120 / img_pil.height
        expected_size = int(expected_scale * img_pil.width), 120

        assert isinstance(img, Image.Image)
        assert img.size == expected_size

    @pytest.mark.parametrize(
        "resample",
        [
            (Image.Resampling.NEAREST),
            (Image.Resampling.BOX),
            (Image.Resampling.BILINEAR),
            (Image.Resampling.HAMMING),
            (Image.Resampling.BICUBIC),
            (Image.Resampling.LANCZOS),
        ],
    )
    def test_resample(self, img_pil: Image.Image, resample: Image.Resampling):
        img = noconvert.scale(img=img_pil, height=120, resample=resample)

        expected_scale = 120 / img_pil.height
        expected_size = int(expected_scale * img_pil.width), 120

        assert isinstance(img, Image.Image)
        assert img.size == expected_size


class TestToIsoRas:
    def test_non_ras_nii(self, nii_3d_non_iso_ras: nib.Nifti1Image):
        img = noconvert.to_ras(nii_3d_non_iso_ras)

        assert isinstance(img, nib.Nifti1Image)
        assert not np.allclose(nii_3d_non_iso_ras.affine, img.affine)


def test_reorient(img_array: np.ndarray):
    reoriented_img = noconvert.reorient(img_array[:, :99, :])
    assert isinstance(reoriented_img, np.ndarray)
    # Assert the sub-array shape is correct before asserting flip
    assert img_array[:, :99, :3].shape == (100, 99, 3)
    assert reoriented_img.shape == (99, 100, 3)
