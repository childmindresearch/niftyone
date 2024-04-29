import numpy as np
import pytest

from niclips.image import apply_affine, coord2ind, ind2coord


@pytest.fixture
def test_affine() -> np.ndarray:
    return np.asarray(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


@pytest.fixture
def test_coord() -> np.ndarray:
    return np.asarray([25.0, 12.3, 17.8])


@pytest.fixture
def test_ind() -> np.ndarray:
    return np.asarray([25, 12, 17])


class TestApplyAffine:
    def test_apply_affine(self, test_affine: np.ndarray, test_coord: np.ndarray):
        coord = apply_affine(test_affine, test_coord)

        assert isinstance(coord, np.ndarray)
        assert coord.shape == test_coord.shape
        assert coord.dtype == test_coord.dtype
        np.testing.assert_allclose(coord, test_coord, rtol=1.0)


class TestCoordInd:
    def test_coord2ind(self, test_affine: np.ndarray, test_ind: np.ndarray):
        coord = ind2coord(test_affine, test_ind)
        ind = coord2ind(test_affine, coord)

        assert isinstance(ind, np.ndarray)
        assert ind.shape == test_ind.shape
        assert ind.dtype == np.int32
        np.testing.assert_allclose(ind, test_ind)

    def test_ind2coord(self, test_affine: np.ndarray, test_coord: np.ndarray):
        ind = coord2ind(test_affine, test_coord)
        coord = ind2coord(test_affine, ind)

        assert isinstance(coord, np.ndarray)
        assert coord.shape == test_coord.shape
        assert coord.dtype == test_coord.dtype
        # Conversion between coordinates and indices will result in a rounding error
        np.testing.assert_allclose(coord, test_coord, rtol=1.0)
