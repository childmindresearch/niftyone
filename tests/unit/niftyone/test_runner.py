from collections.abc import Sequence
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
        mock_generators: Sequence[MagicMock],
        mock_table: BIDSTable,
        tmp_path: Path,
        overwrite: bool,
    ) -> None:
        mock_table.__len__.return_value = 1
        runner = Runner(mock_generators)
        runner.gen_figures(table=mock_table, out_dir=tmp_path, overwrite=overwrite)

        for mock_generator in mock_generators:
            mock_generator.assert_called()

    @pytest.mark.parametrize(
        "table_length, expected_msg", [(0, "Found no images"), (2, "Found 2 images")]
    )
    def test_gen_figures_logging(
        self,
        mock_generators: Sequence[MagicMock],
        mock_table: BIDSTable,
        tmp_path: Path,
        table_length: int,
        expected_msg: str,
        caplog: LogCaptureFixture,
    ):
        mock_table.__len__.return_value = table_length
        runner = Runner(mock_generators)
        runner.gen_figures(table=mock_table, out_dir=tmp_path, overwrite=True)

        assert expected_msg in caplog.text
