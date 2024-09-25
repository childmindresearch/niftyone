import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import fiftyone as fo
import pytest

from niftyone.analysis_levels import launch
from niftyone.metadata import tags

# Disable fiftyone tracking
fo.config.do_not_track = True


@pytest.fixture
def test_config(tmp_path: Path) -> dict[str, str | Path]:
    bids_dir = tmp_path / "bids"
    bids_dir.mkdir(parents=True, exist_ok=True)
    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    return {
        "bids_dir": bids_dir,
        "out_dir": out_dir,
        "ds_name": "test_dataset",
    }


class TestLaunch:
    def test_load_existing(
        self, test_config: dict[str, Any], caplog: pytest.LogCaptureFixture
    ):
        # Setup
        mock_ds_exists = MagicMock(return_value=True)
        mock_load_ds = MagicMock()
        mock_group_tags = MagicMock()
        mock_path_exists = MagicMock(return_value=False)
        mock_launch_app = MagicMock()

        # Call
        with caplog.at_level(logging.INFO):
            with (
                patch("fiftyone.dataset_exists", mock_ds_exists),
                patch("fiftyone.load_dataset", mock_load_ds),
                patch("niftyone.metadata.tags.GroupTags", mock_group_tags),
                patch("pathlib.Path.exists", mock_path_exists),
                patch("fiftyone.launch_app", mock_launch_app),
            ):
                launch(
                    bids_dir=test_config["bids_dir"],
                    out_dir=test_config["out_dir"],
                    ds_name=test_config["ds_name"],
                )

        # Assertions
        mock_ds_exists.assert_called_once_with(test_config["ds_name"])
        mock_load_ds.assert_called_once_with(test_config["ds_name"])
        mock_launch_app.assert_called_once()
        assert "from FiftyOne database" in caplog.text
        assert "Saving QC tags" in caplog.text

    def test_load_directory(
        self, test_config: dict[str, Any], caplog: pytest.LogCaptureFixture
    ):
        # Setup
        mock_ds_exists = MagicMock(return_value=False)
        mock_from_ds_dir = MagicMock(spec=fo.types.FiftyOneDataset)
        mock_group_tags = MagicMock()
        mock_path_exists = MagicMock(return_value=False)
        mock_launch_app = MagicMock()

        # Call
        with caplog.at_level(logging.INFO):
            with (
                patch("fiftyone.dataset_exists", mock_ds_exists),
                patch("fiftyone.Dataset.from_dir", mock_from_ds_dir),
                patch("niftyone.metadata.tags.GroupTags", mock_group_tags),
                patch("pathlib.Path.exists", mock_path_exists),
                patch("fiftyone.launch_app", mock_launch_app),
            ):
                launch(
                    bids_dir=test_config["bids_dir"],
                    out_dir=test_config["out_dir"],
                    ds_name=test_config["ds_name"],
                )

        # Assertions
        mock_ds_exists.assert_called_once_with(test_config["ds_name"])
        mock_from_ds_dir.assert_called_once()
        mock_launch_app.assert_called_once()
        assert f"from {test_config['out_dir'] / 'fiftyone'}" in caplog.text

    def test_load_missing_directory(
        self, test_config: dict[str, Any], caplog: pytest.LogCaptureFixture
    ):
        # Setup
        mock_ds_exists = MagicMock(return_value=False)
        mock_from_ds_dir = MagicMock(side_effect=FileNotFoundError)
        mock_group_tags = MagicMock()
        mock_path_exists = MagicMock(return_value=False)
        mock_launch_app = MagicMock()

        # Call
        with (
            patch("fiftyone.dataset_exists", mock_ds_exists),
            patch("fiftyone.Dataset.from_dir", mock_from_ds_dir),
            patch("niftyone.metadata.tags.GroupTags", mock_group_tags),
            patch("pathlib.Path.exists", mock_path_exists),
            patch("fiftyone.launch_app", mock_launch_app),
        ):
            with pytest.raises(FileNotFoundError, match=".*dataset not found.*"):
                launch(
                    bids_dir=test_config["bids_dir"],
                    out_dir=test_config["out_dir"],
                    ds_name=test_config["ds_name"],
                )

        # Assertions
        mock_ds_exists.assert_called_once_with(test_config["ds_name"])
        assert f"from {test_config['out_dir'] / 'fiftyone'}" in caplog.text

    def test_load_no_dataset_name(
        self, test_config: dict[str, Any], caplog: pytest.LogCaptureFixture
    ):
        # Setup
        mock_ds_exists = MagicMock(return_value=True)
        mock_load_ds = MagicMock()
        mock_group_tags = MagicMock()
        mock_path_exists = MagicMock(return_value=False)
        mock_launch_app = MagicMock()

        # Call
        with caplog.at_level(logging.INFO):
            with (
                patch("fiftyone.dataset_exists", mock_ds_exists),
                patch("fiftyone.load_dataset", mock_load_ds),
                patch("niftyone.metadata.tags.GroupTags", mock_group_tags),
                patch("pathlib.Path.exists", mock_path_exists),
                patch("fiftyone.launch_app", mock_launch_app),
            ):
                launch(
                    bids_dir=test_config["bids_dir"],
                    out_dir=test_config["out_dir"],
                )

        # Assertions
        mock_ds_exists.assert_called_once_with(test_config["bids_dir"].name)
        mock_load_ds.assert_called_once_with(test_config["bids_dir"].name)

    def test_load_qc_key(
        self, test_config: dict[str, Any], caplog: pytest.LogCaptureFixture
    ):
        # Setup
        mock_ds_exists = MagicMock(return_value=True)
        mock_load_ds = MagicMock()
        mock_group_tags = MagicMock()
        mock_path_exists = MagicMock(return_value=False)
        mock_launch_app = MagicMock()

        # Call
        with caplog.at_level(logging.INFO):
            with (
                patch("fiftyone.dataset_exists", mock_ds_exists),
                patch("fiftyone.load_dataset", mock_load_ds),
                patch("niftyone.metadata.tags.GroupTags", mock_group_tags),
                patch("pathlib.Path.exists", mock_path_exists),
                patch("fiftyone.launch_app", mock_launch_app),
            ):
                launch(
                    bids_dir=test_config["bids_dir"],
                    out_dir=test_config["out_dir"],
                    ds_name=test_config["ds_name"],
                    qc_key="QC",
                )

        # Assertions
        mock_ds_exists.assert_called_once_with(f"{test_config['ds_name']}-QC")
        mock_load_ds.assert_called_once_with(f"{test_config['ds_name']}-QC")

    def test_tags_path_exists(
        self, test_config: dict[str, Any], caplog: pytest.LogCaptureFixture
    ):
        # Setup
        mock_ds_exists = MagicMock(return_value=True)
        mock_load_ds = MagicMock()
        mock_group_tags = MagicMock(spec=tags.GroupTags)
        mock_path_exists = MagicMock(return_value=True)
        mock_launch_app = MagicMock()

        # Call
        with caplog.at_level(logging.INFO):
            with (
                patch("fiftyone.dataset_exists", mock_ds_exists),
                patch("fiftyone.load_dataset", mock_load_ds),
                patch("niftyone.metadata.tags.GroupTags.from_json", mock_group_tags),
                patch("pathlib.Path.exists", mock_path_exists),
                patch("fiftyone.launch_app", mock_launch_app),
            ):
                launch(
                    bids_dir=test_config["bids_dir"],
                    out_dir=test_config["out_dir"],
                    ds_name=test_config["ds_name"],
                )

        # Assertions
        mock_group_tags.assert_called_once()
        assert "Loading QC tags" in caplog.text
