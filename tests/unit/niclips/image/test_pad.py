from copy import deepcopy

import numpy as np
import pytest

from niclips.image import Align, pad_to_equal, pad_to_size, pad_to_square


class TestPadToSize:
    @pytest.mark.parametrize("size", [(3), (4)])
    def test_valid_size(self, img_array: np.ndarray, size: int):
        img = pad_to_size(img_array, size=size, axis=2)
        assert isinstance(img, np.ndarray)
        assert img.shape == (100, 100, size)

    def test_invalid_size(self, img_array: np.ndarray):
        with pytest.raises(AssertionError, match=".*too small.*"):
            pad_to_size(img_array, size=1, axis=2)

    @pytest.mark.parametrize(
        "align", [(Align.LEFT), (Align.RIGHT), (Align.TOP), (Align.BOTTOM)]
    )
    def test_diff_align(self, img_array: np.ndarray, align: Align):
        img = pad_to_size(img_array, size=4, axis=2, align=align)
        assert isinstance(img, np.ndarray)


def test_pad_to_equal(img_array: np.ndarray):
    in_imgs = [deepcopy(img_array), deepcopy(img_array)]
    imgs = pad_to_equal(in_imgs, axis=2)

    assert len(imgs) == len(in_imgs)
    assert all([isinstance(img, np.ndarray) for img in imgs])
    assert imgs[0].shape == imgs[1].shape


def test_pad_to_square(img_array: np.ndarray):
    img = pad_to_square(img_array)

    assert isinstance(img, np.ndarray)
    assert img.shape[0] == img.shape[1]
