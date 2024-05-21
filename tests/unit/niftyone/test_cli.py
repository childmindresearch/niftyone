from pathlib import Path
from unittest.mock import patch

import pytest

from niftyone.cli import NiftyOneArgumentParser


@pytest.fixture
def parser() -> NiftyOneArgumentParser:
    return NiftyOneArgumentParser()


class TestParser:
    def test_common_args(self, parser: NiftyOneArgumentParser) -> None:
        with patch(
            "sys.argv",
            [
                "niftyone",
                "bids_dir",
                "out_dir",
                "participant",
                "--overwrite",
                "--verbose",
            ],
        ):
            args = parser.parse_args()
        assert args.bids_dir == Path("bids_dir")
        assert args.out_dir == Path("out_dir")
        assert args.analysis_level == "participant"
        assert args.overwrite
        assert args.verbose

    def test_participant_args(self, parser: NiftyOneArgumentParser) -> None:
        with patch(
            "sys.argv",
            [
                "niftyone",
                "bids_dir",
                "out_dir",
                "participant",
                "--participant-label",
                "01",
                "--index",
                "index.b2t",
                "--qc-dir",
                "qc_dir",
                "--workers",
                "2",
            ],
        ):
            args = parser.parse_args()
        assert args.analysis_level == "participant"
        assert args.participant_label == "01"
        assert args.index == Path("index.b2t")
        assert args.qc_dir == Path("qc_dir")
        assert args.workers == 2

    def test_group_args(self, parser: NiftyOneArgumentParser) -> None:
        with patch(
            "sys.argv",
            ["niftyone", "bids_dir", "out_dir", "group", "--ds-name", "test-ds"],
        ):
            args = parser.parse_args()
        assert args.analysis_level == "group"
        assert args.ds_name == "test-ds"

    def test_launch_args(self, parser: NiftyOneArgumentParser) -> None:
        with patch(
            "sys.argv",
            ["niftyone", "bids_dir", "out_dir", "launch", "--qc-key", "key"],
        ):
            args = parser.parse_args()
        assert args.analysis_level == "launch"
        assert args.qc_key == "key"
