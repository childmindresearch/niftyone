from pathlib import Path
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pkg_resources  # type: ignore[import-untyped]
import pytest
from bids2table import BIDSTable, bids2table
from PIL import Image

plt.set_loglevel("info")


# Fixtures
@pytest.fixture
def mock_img() -> MagicMock:
    mock_img = MagicMock()
    mock_img.ndim = 3
    mock_img.shape = (10, 10, 10)
    mock_img.affine = np.eye(4)
    return mock_img


@pytest.fixture
def img_array() -> np.ndarray:
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def img_pil() -> Image.Image:
    return Image.new("RGB", (100, 100))


@pytest.fixture
def nii_3d_img() -> nib.Nifti1Image:
    img_arr = np.zeros((10, 10, 10))
    img_arr[5, 5, 5] = 1

    return nib.Nifti1Image(dataobj=img_arr, affine=np.eye(4))


@pytest.fixture
def nii_4d_img() -> nib.Nifti1Image:
    img_arr = np.zeros((10, 10, 10, 3))
    img_arr[5, 5, 5, 1] = 10  # Set single voxel in middle volume

    return nib.Nifti1Image(dataobj=img_arr, affine=np.eye(4))


@pytest.fixture
def nii_3d_non_iso_ras() -> nib.Nifti1Image:
    img_arr = np.zeros((10, 10, 10))
    img_arr[5, 5, 5] = 1
    affine = np.array(
        [
            [-2.0, 0, 0, 90.0],
            [0, 3.0, 0.0, -100],
            [0, 0, 4.0, -110.0],
            [0, 0, 0, 1.0],
        ]
    )

    return nib.Nifti1Image(dataobj=img_arr, affine=affine)


@pytest.fixture
def qc_dir() -> Path:
    return Path(pkg_resources.resource_filename(__name__, "../data/ds000102-mriqc"))


@pytest.fixture
def b2t_index(tmp_path: Path) -> BIDSTable | None:
    return bids2table(
        root=pkg_resources.resource_filename(__name__, "../data/ds000102"),
        with_meta=False,
        persistent=True,
        index_path=tmp_path / "b2t",
        workers=1,
    )
