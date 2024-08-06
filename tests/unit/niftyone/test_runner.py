from pathlib import Path
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
def mock_table() -> BIDSTable:
    return MagicMock(spec=BIDSTable)


class TestRunner:
    @pytest.mark.parametrize("overwrite", [(True), (False)])
    def test_gen_figures(
        self,
        mock_generators: list[ViewGenerator],
        mock_table: BIDSTable,
        tmp_path: Path,
        overwrite: bool,
    ) -> None:
        mock_table.filter.return_value = ["f1.nii.gz", "f2.nii.gz"]
        runner = Runner(
            figure_generators=mock_generators,
            out_dir=tmp_path,
            qc_dir=None,
            overwrite=overwrite,
        )
        runner.table = mock_table
        runner.gen_figures()

        for mock_generator in mock_generators:
            mock_generator.assert_called()  # type: ignore [attr-defined]

    @pytest.mark.parametrize(
        "table_return, expected_msg",
        [([], "Found no images"), (["f1.nii.gz", "f2.nii"], "Found 2 images")],
    )
    def test_gen_figures_logging(
        self,
        mock_generators: list[ViewGenerator],
        mock_table: BIDSTable,
        tmp_path: Path,
        table_return: list[str],
        expected_msg: str,
        caplog: LogCaptureFixture,
    ):
        mock_table.filter.return_value = table_return
        runner = Runner(
            figure_generators=mock_generators,
            out_dir=tmp_path,
            qc_dir=None,
            overwrite=True,
        )
        runner.table = mock_table
        runner.gen_figures()

        assert expected_msg in caplog.text

    def test_update_metrics_no_qc_dir(
        self,
        mock_generators: list[ViewGenerator],
        tmp_path: Path,
    ):
        runner = Runner(
            figure_generators=mock_generators,
            out_dir=tmp_path,
            qc_dir=None,
            overwrite=False,
        )
        assert not runner.qc_dir and not runner.update_metrics()  # type: ignore [func-returns-value]

    def test_update_metrics_qc_dir(
        self,
        mock_generators: list[ViewGenerator],
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
            figure_generators=mock_generators,
            out_dir=out_dir.parent,
            qc_dir=qc_dir,
            overwrite=True,
        )
        runner.table = mock_table
        runner.update_metrics()

        assert (out_dir / "sub-01_desc-QCMetrics_T1w.tsv").exists()
