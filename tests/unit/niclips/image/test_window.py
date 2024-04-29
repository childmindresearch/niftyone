import nibabel as nib
import numpy as np

from niclips.image import crop_middle_third, get_fdata, index_img
from niclips.image._window import Window, center_minmax, minmax


class TestMinMax:
    def test_nii(self, nii_3d_img: nib.Nifti1Image):
        window = minmax(img=nii_3d_img)
        data = get_fdata(nii_3d_img)

        assert isinstance(window, Window)
        assert window.vmin == min(data.flatten()) and window.vmax == max(data.flatten())

    def test_arr(self, img_array: np.ndarray):
        img_array[0, 0, 0] = 255.0
        window = minmax(img=img_array)

        assert isinstance(window, Window)
        assert window.vmin == min(img_array.flatten()) and window.vmax == max(
            img_array.flatten()
        )


class TestCenterMinMax:
    def test_3d(self, nii_3d_img: nib.Nifti1Image):
        window = center_minmax(img=nii_3d_img)
        data = crop_middle_third(get_fdata(nii_3d_img))

        assert isinstance(window, Window)
        assert window.vmin == min(data.flatten()) and window.vmax == max(data.flatten())

    # Just test that it will work on a 4D image
    def test_4d(self, nii_4d_img: nib.Nifti1Image):
        window = center_minmax(img=nii_4d_img)
        data = crop_middle_third(get_fdata(index_img(nii_4d_img, idx=None)))

        assert isinstance(window, Window)
        assert window.vmin == min(data.flatten()) and window.vmax == max(data.flatten())
