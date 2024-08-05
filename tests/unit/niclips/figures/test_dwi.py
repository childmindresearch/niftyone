from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from matplotlib.animation import FuncAnimation

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

    img_arr = np.random.rand(10, 10, 10, len(bvals))
    img = nib.Nifti1Image(dataobj=img_arr, affine=np.eye(4))
    img_fpath = tmp_path / "test.nii.gz"
    nib.save(img=img, filename=img_fpath)

    return img_fpath


class TestQSpaceShells:
    @pytest.mark.parametrize("thresh", [(5), (10), (30)])
    def test_default(self, dwi_fpath: Path, thresh: int):
        ani = nodwi.visualize_qspace(dwi=dwi_fpath, thresh=thresh)

        assert isinstance(ani, FuncAnimation)

    def test_save(self, dwi_fpath: Path, tmp_path: Path):
        out_fpath = tmp_path / "test_qspace.mp4"
        nodwi.visualize_qspace(dwi=dwi_fpath, out=out_fpath)

        assert out_fpath.exists()


class TestDwiPerShell:
    def test_default(self, dwi_fpath: Path):
        thresh = 10
        dwis = nodwi.three_view_per_shell(dwi=dwi_fpath, thresh=thresh)
        bval = np.loadtxt(str(dwi_fpath).replace(".nii.gz", ".bval"))

        assert len(dwis) == len(np.unique(nodwi._equate_bvals(bval, thresh=thresh)))
        assert all([isinstance(dwi, nib.Nifti1Image) for dwi in dwis])

    def test_save(self, dwi_fpath: Path, tmp_path: Path):
        thresh = 10
        out_fpath = tmp_path / "test_desc-bval_dwi.mp4"
        bvals = np.loadtxt(str(dwi_fpath).replace(".nii.gz", ".bval"))

        nodwi.three_view_per_shell(dwi=dwi_fpath, out=out_fpath, thresh=10)

        assert all(
            [
                Path(str(out_fpath).replace("bval", f"b{bval}")).exists()
                for bval in np.unique(nodwi._equate_bvals(bvals, thresh=thresh))
            ]
        )

    def test_invalid_shape(self, dwi_fpath: Path):
        img = nib.load(dwi_fpath)
        nib.save(
            nib.Nifti1Image(np.random.rand(10, 10, 10, 5, 2), affine=img.affine),
            filename=dwi_fpath,
        )

        with pytest.raises(ValueError, match=".*wrong shape.*"):
            nodwi.three_view_per_shell(dwi=dwi_fpath, thresh=10)


class TestDwiSignalPerVolume:
    def test_default(self, nii_4d_img: nib.Nifti1Image, tmp_path: Path):
        out_fpath = tmp_path / "test_desc-signalPerVolume_dwi.mp4"

        nodwi.signal_per_volume(dwi=nii_4d_img, out=out_fpath)

        assert out_fpath.exists()
