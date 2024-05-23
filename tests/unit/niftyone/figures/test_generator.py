from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from bids2table import BIDSTable

from niftyone.exceptions import GeneratorError
from niftyone.figures.generator import (
    ViewGenerator,
    create_generator,
    create_generators,
    generator_registry,
    register,
)


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
        entities = {"desc": "test", "ext": ".png"}
        view_fn = None

    return TestGenerator("suffix == 'T1w'", {})


class TestViewGenerator:
    def test_generator_call_generate(
        self, b2t_mock: BIDSTable, test_generator: ViewGenerator, tmp_path: Path
    ) -> None:
        test_generator.generate = MagicMock()
        test_generator(table=b2t_mock, out_dir=tmp_path, overwrite=True)
        test_generator.generate.assert_called()

    def test_generator_no_view_fn(self, test_generator: ViewGenerator) -> None:
        with pytest.raises(ValueError, match="View is not provided.*"):
            test_generator.generate(
                record=MagicMock(), out_dir=MagicMock(spec=Path), overwrite=True
            )


@pytest.fixture
def setup_registry():
    """Fixture to setup and tear down test registry."""
    generator_registry.clear()

    class TestGenerator(ViewGenerator):
        def generate(self, record: pd.Series, out_dir: Path, overwrite: bool) -> None:
            pass

    register("test_view")(TestGenerator)
    yield
    generator_registry.clear()


class TestCreateGenerator:
    @pytest.mark.parametrize(
        "view", [("test_view"), ("test_view(param1='value1', param2=2, param3=.1)")]
    )
    def test_create_generators_view_kwargs(self, setup_registry: Generator, view: str):
        generator = create_generator(view=view, query="suffix == 'T1w'")
        assert isinstance(generator, ViewGenerator)
        assert generator.query == "suffix == 'T1w'"
        assert isinstance(generator.view_kwargs, dict)

    def test_create_generators(self, setup_registry: Generator):
        config = {
            "test1": {"query": "suffix == 'T1w'", "views": ["test_view"]},
            "test2": {"query": "suffix == 'bold'", "views": ["test_view"]},
        }
        generators = create_generators(config)
        assert len(generators) == 2
        assert all(isinstance(generator, ViewGenerator) for generator in generators)
        assert generators[0].query == config["test1"]["query"]
        assert generators[1].query == config["test2"]["query"]

    def test_generator_view_not_found(self):
        config = {"test": {"query": "", "views": ["view1"]}}
<<<<<<< HEAD
        with pytest.raises(KeyError, match=".*not found in registry"):
=======
        with pytest.raises(GeneratorError, match=".*not found in registry"):
>>>>>>> d706e48 (Split create_generator and add kwargs)
            create_generators(config)

    def test_generator_no_views(self):
        config = {"test": {"query": "", "views": []}}
        generators = create_generators(config)
        assert generators == []

    def test_generator_mixed_views(self, setup_registry: Generator):
        config = {
            "test1": {"query": "suffix == 'T1w'", "views": ["test_view"]},
            "test2": {"query": "suffix == 'bold'", "views": ["fake_view"]},
        }

<<<<<<< HEAD
        with pytest.raises(KeyError, match=".*not found in registry"):
=======
        with pytest.raises(GeneratorError, match=".*not found in registry"):
>>>>>>> d706e48 (Split create_generator and add kwargs)
            create_generators(config)
