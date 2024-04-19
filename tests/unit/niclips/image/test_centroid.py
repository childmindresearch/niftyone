import nibabel as nib
import numpy as np
import pytest

from niclips.image import center_of_mass, peak_of_mass


class TestCenterOfMass:
    @pytest.mark.parametrize("mask", [(True), (False)])
    def test_3d(self, nii_3d_img: nib.Nifti1Image, mask: bool):
        centroid = center_of_mass(nii_3d_img, mask=mask)

        assert isinstance(centroid, np.ndarray)
        np.testing.assert_array_almost_equal(centroid, np.asarray([5, 5, 5]))

    @pytest.mark.parametrize("mask", [(True), (False)])
    def test_4d(self, nii_4d_img: nib.Nifti1Image, mask: bool):
        centroid = center_of_mass(nii_4d_img, mask=mask)

        assert isinstance(centroid, np.ndarray)
        np.testing.assert_array_almost_equal(centroid, np.asarray([5, 5, 5]))


class TestPeakOfMass:
    @pytest.mark.parametrize("mask", [(True, False)])
    def test_3d(self, nii_3d_img: nib.Nifti1Image, mask: bool):
        peak = peak_of_mass(nii_3d_img, mask=mask)

        assert isinstance(peak, np.ndarray)
        np.testing.assert_array_almost_equal(peak, np.asarray([5, 5, 5]))

    @pytest.mark.parametrize("mask", [(True, False)])
    def test_4d(self, nii_4d_img: nib.Nifti1Image, mask: bool):
        peak = peak_of_mass(nii_4d_img, mask=mask)

        assert isinstance(peak, np.ndarray)
        np.testing.assert_array_almost_equal(peak, np.asarray([5, 5, 5]))
