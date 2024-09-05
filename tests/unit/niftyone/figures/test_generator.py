from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
from bids2table import BIDSTable

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
    table_mock.ent.query.side_effect = [
        pd.DataFrame({"sub": "01", "ses": "01", "suffix": "T1w"}, index=[0])
    ]

    return table_mock


@pytest.fixture
def test_generator() -> ViewGenerator:
    class TestGenerator(ViewGenerator):
        entities = {"desc": "test", "ext": ".png"}
        view_fn = None

    return TestGenerator(["suffix == 'T1w'"], None, {})


class TestViewGenerator:
    def test_generator_call_generate(
        self, b2t_mock: BIDSTable, test_generator: ViewGenerator, tmp_path: Path
    ) -> None:
        test_generator.generate = MagicMock()  # type: ignore [method-assign]
        test_generator(table=b2t_mock, out_dir=tmp_path, overwrite=True)
        test_generator.generate.assert_called()

    def test_generator_no_view_fn(self, test_generator: ViewGenerator) -> None:
        with pytest.raises(ValueError, match="View is not provided.*"):
            test_generator.generate(
                records=MagicMock(),
                out_dir=MagicMock(spec=Path),
                overwrite=True,
            )


@pytest.fixture
def setup_registry():
    """Fixture to setup and tear down test registry."""
    generator_registry.clear()

    class TestGenerator(ViewGenerator):
        def generate(
            self,
            records: pd.Series,
            out_dir: Path,
            overwrite: bool,
        ) -> None:
            pass

    register("test_view")(TestGenerator)
    yield
    generator_registry.clear()


class TestCreateGenerator:
    @pytest.mark.parametrize(
        ("view", "view_kwargs"),
        [
            ("test_view", {}),
            ("test_view", {"param1": "value1", "param2": 2, "param3": 0.1}),
        ],
    )
    def test_create_generators_view_kwargs(
        self, setup_registry: Generator, view: str, view_kwargs: dict[str, Any]
    ):
        generator = create_generator(
            view=view,
            view_kwargs=view_kwargs,
            join_entities=["sub"],
            queries=["suffix == 'T1w'"],
        )
        assert isinstance(generator, ViewGenerator)
        assert generator.queries == ["suffix == 'T1w'"]
        assert isinstance(generator.view_kwargs, dict)

    def test_create_generators(self, setup_registry: Generator):
        config = {
            "figures": {
                "test1": {"queries": "suffix == 'T1w'", "views": {"test_view": None}},
                "test2": {"queries": "suffix == 'bold'", "views": {"test_view": None}},
            }
        }
        generators = create_generators(config)
        assert len(generators) == 2
        assert all(isinstance(generator, ViewGenerator) for generator in generators)
        assert generators[0].queries == config["figures"]["test1"]["queries"]
        assert generators[1].queries == config["figures"]["test2"]["queries"]

    def test_generator_view_not_found(self):
        config = {"figures": {"test": {"queries": "", "views": {"view1": None}}}}
        with pytest.raises(KeyError, match=".*not found in registry"):
            create_generators(config)

    def test_generator_no_views(self):
        config = {"figures": {"test": {"queries": "", "views": {}}}}
        generators = create_generators(config)
        assert generators == []

    def test_generator_mixed_views(self, setup_registry: Generator):
        config = {
            "figures": {
                "test1": {"queries": "suffix == 'T1w'", "views": {"test_view": None}},
                "test2": {"queries": "suffix == 'bold'", "views": {"fake_view": None}},
            }
        }

        with pytest.raises(KeyError, match=".*not found in registry"):
            create_generators(config)
