from pathlib import Path

import pandas as pd
import pytest
from _pytest.logging import LogCaptureFixture
from bids2table import BIDSEntities, BIDSTable

from niftyone.figures import func


@pytest.fixture
def record_bold(b2t_index: BIDSTable) -> pd.Series:
    return b2t_index.filter_multi(sub="01", run=1, suffix="bold").nested.iloc[0]


class TestBoldRaw:
    def test_bold_default(
        self, record_bold: pd.Series, tmp_path: Path, caplog: LogCaptureFixture
    ):
        out_dir = tmp_path / "out"

        func.raw_bold(record_bold, out_dir=out_dir)

        assert "Processing" in caplog.text
        count = sum(1 for record in caplog.record_tuples if "Generating" in record[2])
        assert count >= 1

    @pytest.mark.parametrize("overwrite", [(False), (True)])
    def test_bold_exists(
        self,
        record_bold: pd.Series,
        tmp_path: Path,
        overwrite: bool,
        caplog: LogCaptureFixture,
    ):
        entities = BIDSEntities.from_dict(record_bold["ent"])

        # Create existing files
        out_dir = tmp_path / "out"
        out_path = entities.with_update(desc="threeViewVideo", ext=".mp4").to_path(
            prefix=out_dir
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.touch()
        out_path = (
            entities.with_update(desc="sliceVideo", ext=".mp4")
            .to_path(prefix=out_dir)
            .touch()
        )
        out_path = (
            entities.with_update(desc="carpet", ext=".png")
            .to_path(prefix=out_dir)
            .touch()
        )
        entities.with_update(desc="meanStd", ext=".png").to_path(prefix=out_dir).touch()

        func.raw_bold(record=record_bold, out_dir=out_dir, overwrite=overwrite)

        if not overwrite:
            assert "Generating" not in caplog.text
        else:
            assert "Generating" in caplog.text

    def test_func_qc(
        self,
        record_bold: pd.Series,
        tmp_path: Path,
        qc_dir: Path,
        caplog: LogCaptureFixture,
    ):
        out_dir = tmp_path / "out"
        func.raw_bold(record_bold, out_dir=out_dir, qc_dir=qc_dir, overwrite=False)

        assert "Processing" in caplog.text
        count = sum(1 for record in caplog.record_tuples if "Generating" in record[2])
        assert count >= 1
