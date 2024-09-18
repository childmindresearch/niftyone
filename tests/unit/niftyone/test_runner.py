from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
from _pytest.logging import LogCaptureFixture
from bids2table import BIDSTable

from niftyone.figures.generator import ViewGenerator
from niftyone.runner import Runner


@pytest.fixture
def mock_generators() -> list[ViewGenerator]:
    return [MagicMock(spec=ViewGenerator) for _ in range(2)]


@pytest.fixture
def mock_config() -> dict[str, Any]:
    return MagicMock(spec=dict[str, Any])


@pytest.fixture
def mock_table() -> BIDSTable:
    return MagicMock(spec=BIDSTable)


class TestRunner:
    @pytest.mark.parametrize("overwrite", [(True), (False)])
    def test_gen_figures(
        self,
        mock_config: dict[str, Any],
        mock_table: BIDSTable,
        tmp_path: Path,
        overwrite: bool,
        caplog: LogCaptureFixture,
    ) -> None:
        mock_table.filter.return_value = ["f1.nii.gz", "f2.nii.gz"]
        runner = Runner(
            out_dir=tmp_path,
            qc_dir=None,
            overwrite=overwrite,
            config=mock_config,
        )
        runner.table = mock_table
        runner.gen_figures()

    @pytest.mark.parametrize(
        "table_return, expected_msg",
        [([], "Found no images"), (["f1.nii.gz", "f2.nii"], "Found 2 images")],
    )
    def test_gen_figures_logging(
        self,
        mock_config: dict[str, Any],
        mock_table: BIDSTable,
        tmp_path: Path,
        table_return: list[str],
        expected_msg: str,
        caplog: LogCaptureFixture,
    ):
        mock_table.filter.return_value = table_return
        runner = Runner(
            out_dir=tmp_path,
            qc_dir=None,
            overwrite=True,
            config=mock_config,
        )
        runner.table = mock_table
        runner.gen_figures()

        assert expected_msg in caplog.text

    def test_update_metrics_no_qc_dir(
        self,
        mock_config: dict[str, Any],
        tmp_path: Path,
    ):
        runner = Runner(
            out_dir=tmp_path,
            qc_dir=None,
            overwrite=False,
            config=mock_config,
        )
        assert not runner.qc_dir and not runner.update_metrics()  # type: ignore [func-returns-value]

    def test_update_metrics_qc_dir(
        self,
        mock_config: dict[str, Any],
        mock_table: BIDSTable,
        tmp_path: Path,
    ):
        # Create mock qc_file
        (out_dir := (tmp_path / "out" / "sub-01")).mkdir(parents=True, exist_ok=True)
        (qc_dir := (tmp_path / "qc_dir")).mkdir(parents=True, exist_ok=True)
        qc_df = pd.DataFrame(data={"bids_name": ["sub-01_T1w"], "fake_metric": ["1.0"]})
        qc_df.to_csv(qc_dir / "group_T1w.tsv", sep="\t", index=False)

        mock_table.filter.return_value.nested.iterrows.return_value = [
            (
                None,
                pd.DataFrame(
                    data={"ent": {"sub": "01", "suffix": "T1w", "ext": ".nii.gz"}}
                ),
            )
        ]
        runner = Runner(
            out_dir=out_dir.parent,
            qc_dir=qc_dir,
            overwrite=True,
            config=mock_config,
        )
        runner.table = mock_table
        runner.update_metrics()

        assert (out_dir / "sub-01_metrics-QCMetrics_T1w.tsv").exists()
