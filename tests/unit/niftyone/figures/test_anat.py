from pathlib import Path

import pandas as pd
import pytest
from _pytest.logging import LogCaptureFixture
from bids2table import BIDSEntities, BIDSTable

from niftyone.figures import anat


@pytest.fixture
def record_t1w(b2t_index: BIDSTable) -> pd.Series:
    return b2t_index.filter_multi(sub="01", suffix="T1w").nested.iloc[0]


class TestAnatRaw:
    def test_t1w_default(
        self, record_t1w: pd.Series, tmp_path: Path, caplog: LogCaptureFixture
    ):
        out_dir = tmp_path / "out"

        anat.raw_t1w(record_t1w, out_dir=out_dir)

        assert "Processing" in caplog.text
        assert "Generating" in caplog.text

    @pytest.mark.parametrize("overwrite", [(False), (True)])
    def test_t1w_exists(
        self,
        record_t1w: pd.Series,
        tmp_path: Path,
        overwrite: bool,
        caplog: LogCaptureFixture,
    ):
        entities = BIDSEntities.from_dict(record_t1w["ent"])

        # Create existing files
        out_dir = tmp_path / "out"
        out_path = entities.with_update(desc="threeView", ext=".png").to_path(
            prefix=out_dir
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.touch()
        out_path = (
            entities.with_update(desc="sliceVideo", ext=".mp4")
            .to_path(prefix=out_dir)
            .touch()
        )

        anat.raw_t1w(record=record_t1w, out_dir=out_dir, overwrite=overwrite)

        if not overwrite:
            assert "Generating" not in caplog.text
        else:
            assert "Generating" in caplog.text

    def test_t1w_qc(
        self,
        record_t1w: pd.Series,
        tmp_path: Path,
        qc_dir: Path,
        caplog: LogCaptureFixture,
    ):
        out_dir = tmp_path / "out"
        anat.raw_t1w(record_t1w, out_dir=out_dir, qc_dir=qc_dir, overwrite=False)

        assert "Processing" in caplog.text
        assert "Generating" in caplog.text
