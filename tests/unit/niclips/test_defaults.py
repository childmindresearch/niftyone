from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import niclips.defaults as nodefaults


def test_get_default_coord(mock_img: MagicMock):
    mock_peak_mass = MagicMock(return_value=np.array([1.0, 2.0, 3.0]))
    with patch("niclips.image.peak_of_mass", mock_peak_mass):
        default_coord = nodefaults.get_default_coord(mock_img)
    assert default_coord == (2.0, 2.0, 3.0)


def test_get_default_window(mock_img: MagicMock):
    mock_center_minmax = MagicMock(return_value=(100.0, 1000.0))
    with patch("niclips.image.center_minmax", mock_center_minmax):
        default_window = nodefaults.get_default_window(mock_img)
    assert default_window == (100.0, 1000.0)


@pytest.fixture
def mock_window() -> MagicMock:
    mock_window = MagicMock()
    mock_window.vmin = 100.0
    mock_window.vmax = 1000.0
    return mock_window


class TestGetDefaultVminVmax:
    def test_no_override(self, mock_img: MagicMock, mock_window: MagicMock):
        mock_default_window = MagicMock(return_value=mock_window)
        with patch("niclips.defaults.get_default_window", mock_default_window):
            vmin, vmax = nodefaults.get_default_vmin_vmax(mock_img)
        assert vmin == mock_window.vmin
        assert vmax == mock_window.vmax

    def test_override(self, mock_img: MagicMock, mock_window: MagicMock):
        mock_default_window = MagicMock(return_value=mock_window)
        with patch("niclips.defaults.get_default_window", mock_default_window):
            vmin, vmax = nodefaults.get_default_vmin_vmax(
                mock_img, vmin=10.0, vmax=100.0
            )
        assert vmin == 10.0 and vmin != mock_window.vmin
        assert vmax == 100.0 and vmax != mock_window.vmax
