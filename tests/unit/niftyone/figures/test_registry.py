from collections.abc import Generator
from pathlib import Path

import pandas as pd
import pytest

from niftyone.figures.generator import ViewGenerator
from niftyone.figures.registry import (
    clear_registry,
    create_generator,
    list_generators,
    register_generator,
)


@pytest.fixture
def setup_registry():
    """Fixture to setup and tear down test registry."""
    clear_registry()

    class TestGenerator(ViewGenerator):
        def generate(self, record: pd.Series, out_dir: Path, overwrite: bool) -> None:
            pass

    register_generator("test_view", TestGenerator)
    yield
    clear_registry()


class TestCreateGenerator:
    def test_create_generator(self, setup_registry: Generator):
        generator = create_generator(
            "test_view", query="suffix == 'T1w'", view_kwargs={}
        )
        assert isinstance(generator, ViewGenerator)
        assert generator.query == "suffix == 'T1w'"

    def test_list_generators(self, setup_registry: Generator):
        available_views = list_generators()
        assert available_views == ["test_view"]

    def test_generator_view_not_found(self, setup_registry: Generator):
        with pytest.raises(KeyError, match=".*not found in registry"):
            create_generator("non_existent_view", query="suffix == 'T1w'")
