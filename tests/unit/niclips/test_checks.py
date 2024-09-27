from unittest.mock import MagicMock

import numpy as np
import pytest

import niclips.checks as nichecks


class TestCheck3d:
    def test_fail(self, mock_img: MagicMock):
        mock_img.ndim = 4
        with pytest.raises(ValueError, match="Expected 3d image.*"):
            nichecks.check_3d(mock_img)

    def test_pass(self, mock_img: MagicMock):
        nichecks.check_3d(mock_img)


class TestCheck4d:
    def test_fail(self, mock_img: MagicMock):
        with pytest.raises(ValueError, match="Expected 4d image.*"):
            nichecks.check_4d(mock_img)

    def test_pass(self, mock_img: MagicMock):
        mock_img.ndim = 4
        nichecks.check_4d(mock_img)


class TestCheck3d4d:
    def test_fail(self, mock_img: MagicMock):
        mock_img.ndim = 5
        with pytest.raises(ValueError, match=".*3d or 4d.*"):
            nichecks.check_3d_4d(mock_img)

    @pytest.mark.parametrize("ndim", [(3), (4)])
    def test_pass(self, mock_img: MagicMock, ndim: int):
        mock_img.ndim = ndim
        nichecks.check_3d_4d(mock_img)


class TestCheckAtMost4D:
    def test_fail(self, mock_img: MagicMock):
        mock_img.ndim = 5
        with pytest.raises(ValueError, match=".*at most 4d.*"):
            nichecks.check_atmost_4d(mock_img)

    @pytest.mark.parametrize("ndim", [(2), (3), (4)])
    def test_pass(self, mock_img: MagicMock, ndim: int):
        mock_img.ndim = ndim
        nichecks.check_atmost_4d(mock_img)


class TestCheckRAS:
    def test_fail(self, mock_img: MagicMock):
        mock_img.affine[0] = [0, 1, 0, 0]
        mock_img.affine[1] = [1, 0, 0, 0]
        with pytest.raises(ValueError, match=".*RAS orientation.*"):
            nichecks.check_ras(mock_img)

    def test_pass(self, mock_img: MagicMock):
        nichecks.check_ras(mock_img)


class TestCheckIsoRAS:
    def test_fail(self, mock_img: MagicMock):
        mock_img.affine = np.diag([np.random.randint(1, 11) for _ in range(4)])
        with pytest.raises(ValueError, match=".*isotropic voxels.*"):
            nichecks.check_ras(mock_img)

    def test_pass(self, mock_img: MagicMock):
        nichecks.check_ras(mock_img)
