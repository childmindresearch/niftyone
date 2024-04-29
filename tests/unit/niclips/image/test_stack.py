import numpy as np
import pytest

from niclips.image._stack import image_grid, stack_images


class TestImageGrid:
    def test_single(self, img_array: np.ndarray):
        grid = image_grid([img_array], pad=0)
        assert isinstance(grid, np.ndarray)
        assert grid.shape == img_array.shape

    def test_multiple(self, img_array: np.ndarray):
        img2_array = np.ones((120, 120, 3), dtype=np.uint8)
        grid = image_grid([img_array, img2_array], nrows=2, pad=0)
        assert isinstance(grid, np.ndarray)
        assert grid.shape == (220, 120, 3)
        np.testing.assert_array_equal(grid[:100, :100, :], img_array)
        np.testing.assert_array_equal(grid[100:, 100:, :], img2_array[:, :20, :])

    def test_pad_and_fill(self, img_array: np.ndarray):
        img2_array = np.ones((120, 120, 3), dtype=np.uint8)
        grid = image_grid([img_array, img2_array], nrows=2, pad=2, fill_value=100)
        assert isinstance(grid, np.ndarray)
        assert grid.shape == (228, 124, 3)
        assert np.all(grid[:104, 104:, :] == 100)


# Only need to test for assertions
class TestStackImages:
    def test_diff_ndims(self, img_array: np.ndarray):
        img2_array = np.zeros((1, 1, 1, 1), dtype=np.uint8)
        with pytest.raises(AssertionError, match=".*different ndims"):
            stack_images([img_array, img2_array])

    def test_invalid_ndims(self):
        img_arr = np.zeros((1, 1, 1, 1), dtype=np.uint8)
        img2_arr = np.zeros((1, 1, 1, 2), dtype=np.uint8)

        with pytest.raises(AssertionError, match=".*2 or 3 dims"):
            stack_images([img_arr, img2_arr])

    def test_invalid_axis(self, img_array: np.ndarray):
        with pytest.raises(AssertionError, match="Invalid axis.*"):
            stack_images([img_array], axis=2)
