from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pandas import DataFrame

from niftyone.metadata.tags import TAGS, GroupTags


@pytest.fixture
def test_dataset() -> MagicMock:
    label1_mock = MagicMock(label="dummy", tags=TAGS)

    sample1_mock = MagicMock()
    sample1_mock.__getitem__.return_value = label1_mock

    dataset_mock = MagicMock()
    dataset_mock.__iter__.return_value = [sample1_mock]

    return dataset_mock


@pytest.fixture
def test_dataset_extras() -> MagicMock:
    new_tags = deepcopy(TAGS)
    new_tags.append("Dummy Tag")
    label1_mock = MagicMock(label="dummy", tags=new_tags)

    sample1_mock = MagicMock()
    sample1_mock.__getitem__.return_value = label1_mock

    dataset_mock = MagicMock()
    dataset_mock.__iter__.return_value = [sample1_mock]

    return dataset_mock


@pytest.fixture
def test_group_tags(test_dataset: MagicMock) -> GroupTags:
    return GroupTags.from_dataset(test_dataset)


class TestFromDataset:
    def test_from_dataset_all_tags(self, test_dataset: MagicMock):
        group_tags = GroupTags.from_dataset(test_dataset)

        expected_tags_dict: dict[str, Any] = {
            "dummy": {key: True for key in TAGS},
        }
        expected_tags_dict["dummy"]["_Extra"] = []

        assert isinstance(group_tags, GroupTags)
        assert group_tags.tags_dict == expected_tags_dict

    def test_from_dataset_extras(self, test_dataset_extras: MagicMock):
        group_tags = GroupTags.from_dataset(test_dataset_extras)

        expected_tags_dict: dict[str, Any] = {
            "dummy": {key: True for key in TAGS},
        }
        expected_tags_dict["dummy"]["_Extra"] = ["Dummy Tag"]

        assert isinstance(group_tags, GroupTags)
        assert group_tags.tags_dict == expected_tags_dict


class TestDataframes:
    def test_to_dataframe(self, test_group_tags: GroupTags):
        group_tags_df = test_group_tags.to_df()

        assert isinstance(group_tags_df, DataFrame)
        assert group_tags_df.shape == (1, len(TAGS) + 1)

    def test_from_dataframe(self, test_group_tags: GroupTags):
        group_tags_df = test_group_tags.to_df()

        new_group_tags = GroupTags.from_df(group_tags_df)

        assert isinstance(new_group_tags, GroupTags)
        assert new_group_tags.equals(test_group_tags)


class TestCSVs:
    def test_to_csv(self, test_group_tags: GroupTags, tmp_path: Path):
        csv_path = tmp_path / "test.csv"
        test_group_tags.to_csv(csv_path)

        assert csv_path.exists()

    def test_from_csv(self, test_group_tags: GroupTags, tmp_path: Path):
        csv_path = tmp_path / "test.csv"
        test_group_tags.to_csv(csv_path)

        new_group_tags = GroupTags.from_csv(csv_path)

        assert isinstance(new_group_tags, GroupTags)
        assert new_group_tags.equals(test_group_tags)


class TestJSONs:
    def test_to_json(self, test_group_tags: GroupTags, tmp_path: Path):
        json_path = tmp_path / "test.json"
        test_group_tags.to_json(json_path)

        assert json_path.exists()

    def test_from_json(self, test_group_tags: GroupTags, tmp_path: Path):
        json_path = tmp_path / "test.json"
        test_group_tags.to_json(json_path)

        new_group_tags = GroupTags.from_json(json_path)

        assert isinstance(new_group_tags, GroupTags)
        assert new_group_tags.equals(test_group_tags)


class TestEquals:
    def test_true(self, test_group_tags: GroupTags):
        other_group_tags = deepcopy(test_group_tags)

        assert test_group_tags.equals(other_group_tags) is True

    def test_false(self, test_group_tags: GroupTags, test_dataset_extras: MagicMock):
        other_group_tags = GroupTags.from_dataset(test_dataset_extras)

        assert test_group_tags.equals(other_group_tags) is False


class TestApply:
    def test_apply_standard_tags(
        self, test_dataset: MagicMock, test_group_tags: GroupTags
    ):
        label_mock = MagicMock(label="dummy", tags=[])
        sample_mock = MagicMock()
        sample_mock.__getitem__.return_value = label_mock
        dataset_mock = MagicMock()
        dataset_mock.__iter__.return_value = [sample_mock]

        test_group_tags.apply(dataset_mock)
        new_group_tags = GroupTags.from_dataset(dataset_mock)

        assert new_group_tags.equals(test_group_tags)

    def test_apply_extra_tags(self, test_dataset_extras: MagicMock):
        test_group_tags = GroupTags.from_dataset(test_dataset_extras)
        label_mock = MagicMock(label="dummy", tags=[])
        sample_mock = MagicMock()
        sample_mock.__getitem__.return_value = label_mock
        dataset_mock = MagicMock()
        dataset_mock.__iter__.return_value = [sample_mock]

        test_group_tags.apply(dataset_mock)
        new_group_tags = GroupTags.from_dataset(dataset_mock)

        assert new_group_tags.equals(test_group_tags)
