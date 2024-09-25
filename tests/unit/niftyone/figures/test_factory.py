from collections.abc import Generator, Mapping
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
from bids2table import BIDSTable

from niftyone.figures.factory import (
    View,
    create_view,
    create_views,
    register,
    view_registry,
)


@pytest.fixture
def b2t_mock():
    table_mock = MagicMock(spec=BIDSTable)
    table_mock.ent.query.side_effect = [
        pd.DataFrame({"sub": "01", "ses": "01", "suffix": "T1w"}, index=[0])
    ]

    return table_mock


@pytest.fixture
def test_view() -> View:
    class TestGenerator(View):
        entities = {"desc": "test", "ext": ".png"}
        view_fn = None

    return TestGenerator(["suffix == 'T1w'"], None, {})


class TestViewFactory:
    def test_view_call_create(
        self, b2t_mock: BIDSTable, test_view: View, tmp_path: Path
    ) -> None:
        test_view.create = MagicMock()  # type: ignore [method-assign]
        test_view(table=b2t_mock, out_dir=tmp_path, overwrite=True)
        test_view.create.assert_called()

    def test_view_no_view_fn(self, test_view: View) -> None:
        with pytest.raises(ValueError, match=".*unable to create view.*"):
            test_view.create(
                records=MagicMock(),
                out_dir=MagicMock(spec=Path),
                overwrite=True,
            )


@pytest.fixture
def setup_registry():
    """Fixture to setup and tear down test registry."""
    view_registry.clear()

    class TestView(View):
        def create(
            self,
            records: pd.Series,
            out_dir: Path,
            overwrite: bool,
        ) -> None:
            pass

    register("test_view")(TestView)
    yield
    view_registry.clear()


class TestCreateFactory:
    @pytest.mark.parametrize(
        ("view", "view_kwargs"),
        [
            ("test_view", {}),
            ("test_view", {"param1": "value1", "param2": 2, "param3": 0.1}),
        ],
    )
    def test_create_views_view_kwargs(
        self, setup_registry: View, view: str, view_kwargs: dict[str, Any]
    ):
        factory = create_view(
            view=view,
            view_kwargs=view_kwargs,
            join_entities=["sub"],
            queries=["suffix == 'T1w'"],
        )
        assert isinstance(factory, View)
        assert factory.queries == ["suffix == 'T1w'"]
        assert isinstance(factory.view_kwargs, Mapping)

    def test_create_views(self, setup_registry: View):
        config = {
            "figures": {
                "test1": {"queries": "suffix == 'T1w'", "views": {"test_view": None}},
                "test2": {"queries": "suffix == 'bold'", "views": {"test_view": None}},
            }
        }
        views = create_views(config)
        assert len(views) == 2
        assert all(isinstance(factory, View) for factory in views)
        assert views[0].queries == config["figures"]["test1"]["queries"]
        assert views[1].queries == config["figures"]["test2"]["queries"]

    def test_view_view_not_found(self):
        config = {"figures": {"test": {"queries": "", "views": {"view1": None}}}}
        with pytest.raises(KeyError, match=".*not found in registry"):
            create_views(config)

    def test_view_no_views(self):
        config = {"figures": {"test": {"queries": "", "views": {}}}}
        views = create_views(config)
        assert views == []

    def test_view_mixed_views(self, setup_registry: Generator):
        config = {
            "figures": {
                "test1": {"queries": "suffix == 'T1w'", "views": {"test_view": None}},
                "test2": {"queries": "suffix == 'bold'", "views": {"fake_view": None}},
            }
        }

        with pytest.raises(KeyError, match=".*not found in registry"):
            create_views(config)
