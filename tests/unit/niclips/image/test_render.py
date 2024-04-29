from unittest.mock import MagicMock, patch

import nibabel as nib
import pytest
from PIL import Image

from niclips.image._render import render_slice


class TestRenderSlice:
    def test_no_options(self, nii_3d_img: nib.Nifti1Image):
        # Setup
        mock_scale = MagicMock()
        mock_draw_annotation = MagicMock()

        with (
            patch("niclips.image._convert.scale", mock_scale),
            patch("niclips.image.annotate", mock_draw_annotation),
        ):
            img = render_slice(
                img=nii_3d_img, axis=0, coord=(0, 0, 0), height=None, annotate=False
            )

        assert isinstance(img, Image.Image)
        mock_scale.assert_not_called()
        mock_draw_annotation.assert_not_called()

    @pytest.mark.parametrize("axis", [(0), (1), (2)])
    def test_annotate_height(self, nii_3d_img: nib.Nifti1Image, axis: int):
        img = render_slice(
            img=nii_3d_img, axis=axis, coord=(0, 0, 0), height=256, annotate=True
        )

        assert isinstance(img, Image.Image)
