from pathlib import Path

import numpy as np
import pytest

from niclips.figures import dwi as nodwi


@pytest.fixture
def dwi_fpath(tmp_path: Path) -> Path:
    bvals = np.array([5, 1600, 1600, 1600, 1595, 1600, 1605, 1595, 1600, 1595, 1595])
    bval_fpath = tmp_path / "test.bval"
    np.savetxt(bval_fpath, bvals)

    bvecs = np.array(
        [
            [
                0.57735026,
                -0.9999969,
                -0.586878,
                -0.8942548,
                -0.72977316,
                0.15373544,
                -0.11708178,
                0.40867075,
                -0.21039648,
                -0.09445284,
                0.13537486,
            ],
            [
                0.57735032,
                0.0,
                0.71746105,
                -0.44684735,
                -0.54764342,
                0.21455349,
                0.62163621,
                0.81375086,
                -0.62104523,
                -0.22652406,
                0.99029183,
            ],
            [
                -0.57735038,
                0.00250311,
                -0.37526515,
                0.02521372,
                -0.40928939,
                0.96453744,
                0.77450669,
                -0.41327715,
                0.75500739,
                -0.96941489,
                -0.03155597,
            ],
        ]
    )
    bvec_fpath = tmp_path / "test.bvec"
    np.savetxt(bvec_fpath, bvecs)

    return tmp_path / "test.nii.gz"


class TestQSpaceShells:
    @pytest.mark.parametrize("thresh", [(5), (10), (30)])
    def test_default(self, dwi_fpath: Path, thresh: int):
        nodwi.visualize_qspace(dwi=dwi_fpath, thresh=thresh)

    def test_save(self, dwi_fpath: Path, tmp_path: Path):
        out_fpath = tmp_path / "test_qspace.mp4"
        nodwi.visualize_qspace(dwi=dwi_fpath, out=out_fpath)

        assert out_fpath.exists()
