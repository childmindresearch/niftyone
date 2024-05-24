from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from bids2table import BIDSTable

from niftyone.figures.generator import ViewGenerator


@pytest.fixture
def b2t_mock():
    table_mock = MagicMock(spec=BIDSTable)
    table_mock.ent.query.return_value.index = [0]
    table_mock.nested.loc.__get__item.side_effect = [
        pd.Series(
            {"ent": {"sub": "01", "suffix": "T1w"}, "finfo": {"file_path": "path1.nii"}}
        ),
    ]

    return table_mock


@pytest.fixture
def test_generator() -> ViewGenerator:
    class TestGenerator(ViewGenerator):
        def generate(self, record: pd.Series, out_dir: Path, overwrite: bool) -> None:
            pass

    return TestGenerator("suffix == 'T1w'", {})


class TestViewGenerator:
    def test_generator_call_generate(
        self, b2t_mock: BIDSTable, test_generator: ViewGenerator, tmp_path: Path
    ) -> None:
        test_generator.generate = MagicMock()
        test_generator(table=b2t_mock, out_dir=tmp_path, overwrite=True)
        test_generator.generate.assert_called()

    def test_generator_call_generate_common(
        self, b2t_mock: BIDSTable, test_generator: ViewGenerator, tmp_path: Path
    ) -> None:
        with (
            patch("nibabel.nifti1.load", return_value=MagicMock()),
            patch("niclips.image.to_iso_ras", return_value=MagicMock()),
            patch("bids2table.BIDSEntities.from_dict", return_value=MagicMock()),
        ):
            test_generator.generate = MagicMock()
            test_generator.generate_common(
                record=b2t_mock.nested.loc[0],
                out_dir=tmp_path,
                overwrite=True,
                desc="test",
                ext=".png",
                view_fn=lambda img, out_path: None,
            )

            test_generator.generate.assert_not_called()
