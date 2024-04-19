import nibabel as nib
import numpy as np
import pytest

from niclips.image._slice import (
    crop_middle_third,
    index_img,
    slice_array,
    slice_volume,
)
from niclips.typing import NiftiLike


class TestSliceVolume:
    def test_img_3d_nii(self, nii_3d_img: nib.Nifti1Image):
        slc = slice_volume(nii_3d_img)
        assert isinstance(slc, np.ndarray)

    def test_img_4d_nii(self, nii_4d_img: nib.Nifti1Image):
        slc = slice_volume(nii_4d_img)
        assert isinstance(slc, np.ndarray)

    def test_img_arr(self, img_array: np.ndarray):
        slc = slice_volume(img_array)
        assert isinstance(slc, np.ndarray)


class TestIndexImg:
    @pytest.mark.parametrize("idx", [(None), (0)])
    def test_index_4d_nii(self, nii_4d_img: nib.Nifti1Image, idx: int | None):
        slc = index_img(nii_4d_img, idx=idx)
        assert isinstance(slc, NiftiLike)

    def test_index_3d_nii(self, nii_3d_img: nib.Nifti1Image):
        with pytest.raises(ValueError):
            index_img(nii_3d_img)


class TestCropMiddleThird:
    @pytest.mark.parametrize("axis", [(None), ((0, 1, 2))])
    def test_crop_middle_third(
        self, img_array: np.ndarray, axis: int | tuple[int, ...] | None
    ):
        if axis is None:
            cropped = crop_middle_third(img_array)
            assert cropped.shape[0] == img_array.shape[0] // 3
        else:
            cropped = crop_middle_third(img_array, axis=axis)
            assert isinstance(axis, tuple)
            for idx in range(len(axis)):
                assert cropped.shape[idx] == img_array.shape[idx] // 3
        assert isinstance(cropped, np.ndarray)


class TestSliceArray:
    @pytest.mark.parametrize("axis", [(0), (1), (2)])
    def test_int_idx(self, img_array: np.ndarray, axis: int):
        expected_arr: dict[int, np.ndarray] = {
            0: img_array[0, :, :],
            1: img_array[:, 0, :],
            2: img_array[:, :, 0],
        }

        arr = slice_array(img_array, 0, axis=axis)

        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, expected_arr.get(axis))

    @pytest.mark.parametrize("axis", [(0), (1), (2)])
    def test_slice_idx(self, img_array: np.ndarray, axis: int):
        expected_arr: dict[int, np.ndarray] = {
            0: img_array[:2, :, :],
            1: img_array[:, :2, :],
            2: img_array[:, :, :2],
        }
        arr = slice_array(img_array, idx=slice(2), axis=axis)

        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, expected_arr.get(axis))
