from pathlib import Path
from unittest.mock import MagicMock

import pytest
from _pytest.logging import LogCaptureFixture
from bids2table import BIDSTable

from niftyone.figures.generator import ViewGenerator
from niftyone.runner import Runner


@pytest.fixture
def mock_generators() -> list[MagicMock]:
    return [MagicMock(spec=ViewGenerator) for _ in range(2)]


@pytest.fixture
def mock_table() -> MagicMock:
    return MagicMock(spec=BIDSTable)


class TestRunner:
    @pytest.mark.parametrize("overwrite", [(True), (False)])
    def test_gen_figures(
        self,
        mock_generators: list[MagicMock],
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
            mock_generator.assert_called()

    @pytest.mark.parametrize(
        "table_return, expected_msg",
        [([], "Found no images"), (["f1.nii.gz", "f2.nii"], "Found 2 images")],
    )
    def test_gen_figures_logging(
        self,
        mock_generators: list[MagicMock],
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
