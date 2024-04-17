"""Test import/export of dataset tags."""

from pathlib import Path

import fiftyone as fo
import pytest
from PIL import Image

from niftyone.tags import TAGS, GroupTags


@pytest.fixture()
def dummy_tag_ds(tmp_path: Path) -> fo.Dataset:
    """Dataset with some dummy tags."""
    dataset: fo.Dataset = fo.Dataset("dummy_tag")

    dummy_path = tmp_path / "dummy.png"
    Image.new("RGB", (100, 100)).save(dummy_path)

    samples = []

    for ii in range(10):
        sample = fo.Sample(filepath=dummy_path)
        label = fo.Classification(label=f"dummy-{ii}")
        label.tags = TAGS[:ii]
        label.tags.append("__DUMMY__")
        sample["group_key"] = label
        samples.append(sample)

    dataset.add_samples(samples)
    return dataset


def test_group_tags(dummy_tag_ds: fo.Dataset, tmp_path: Path) -> None:
    """Test import/export of dataset tags."""
    # test from_dataset
    group_tags = GroupTags.from_dataset(dummy_tag_ds)
    assert len(group_tags.tags_dict) == len(dummy_tag_ds)

    # test df conversion
    group_tags_df = group_tags.to_df()
    assert group_tags_df.shape == (len(dummy_tag_ds), len(TAGS) + 1)

    group_tags2 = GroupTags.from_df(group_tags_df)
    assert group_tags.equals(group_tags2)

    # test csv conversion
    group_tags.to_csv(tmp_path / "group_tags.csv")
    group_tags2 = GroupTags.from_csv(tmp_path / "group_tags.csv")
    assert group_tags.equals(group_tags2)

    # test json conversion
    group_tags.to_json(tmp_path / "group_tags.json")
    group_tags2 = GroupTags.from_json(tmp_path / "group_tags.json")
    assert group_tags.equals(group_tags2)

    # test apply to new purged dataset
    dummy_tag_ds2 = dummy_tag_ds.clone(name="dummy_tag2")
    for sample in dummy_tag_ds2:
        sample["group_key"].tags = []

    group_tags.apply(dummy_tag_ds2)
    group_tags2 = GroupTags.from_dataset(dummy_tag_ds2)
    assert group_tags.equals(group_tags2)


if __name__ == "__main__":
    pytest.main([__file__])
