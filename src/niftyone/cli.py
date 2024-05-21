"""CLI-related functions."""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Sequence


class NiftyOneArgumentParser:
    """NiftyOne CLI parser."""

    def __init__(self) -> None:
        self.parser = ArgumentParser(
            prog="niftyone",
            usage="%(prog)s bids_dir output_dir analysis_level [options]",
            description="""
            NiftyOne is a comphrensive tool designed to aid
            large-scale QC of BIDS datasets through visualization
            and quantitative metrics.
            """,
        )
        self._add_common_args()
        self._add_participant_args()
        self._add_group_args()
        self._add_launch_args()

    def _add_common_args(self) -> None:
        """Common (non-analysis specific) arguments."""
        self.parser.add_argument(
            "bids_dir",
            metavar="bids_dir",
            type=Path,
            help="Path to BIDS dataset",
        )
        self.parser.add_argument(
            "out_dir",
            metavar="output_dir",
            type=Path,
            help="Path to output directory",
        )
        self.parser.add_argument(
            "analysis_level",
            metavar="analysis_level",
            type=str,
            choices=["participant", "group", "launch"],
            help="Analysis level",
        )
        self.parser.add_argument(
            "--overwrite",
            "-x",
            help="Overwrite previous results",
            action="store_true",
        )
        self.parser.add_argument(
            "--verbose", "-v", help="Verbose logging.", action="store_true"
        )

    def _add_participant_args(self) -> None:
        """Participant-level CLI arguments."""
        self.participant_level = self.parser.add_argument_group(
            "Participant level options"
        )
        self.participant_level.add_argument(
            "--participant-label",
            "--sub",
            metavar="LABEL",
            type=str,
            default=None,
            help="Participant to analyze (default: all)",
        )
        self.participant_level.add_argument(
            "--index",
            metavar="PATH",
            type=Path,
            default=None,
            help="Pre-computed bids2table index path (default: {bids_dir}/index.b2t)",
        )
        self.participant_level.add_argument(
            "--qc-dir",
            metavar="PATH",
            type=Path,
            default=None,
            help=(
                "Path to pre-computed QC outputs "
                "(default: {bids_dir}/derivatives/mriqc)"
            ),
        )
        self.participant_level.add_argument(
            "--workers",
            "-w",
            metavar="COUNT",
            type=int,
            help="Number of worker processes. Setting to -1 runs as many processes as "
            "there are cores available. (default: 1)",
            default=1,
        )

    def _add_group_args(self) -> None:
        """Group-level CLI arguments."""
        self.group_level = self.parser.add_argument_group("Group level options")
        self.group_level.add_argument(
            "--ds-name",
            metavar="DATASET",
            type=str,
            default=None,
            help="Name of NiftyOne dataset.",
        )

    def _add_launch_args(self) -> None:
        """Application launch CLI arguments."""
        self.launch_group = self.parser.add_argument_group("Launch level options")
        self.launch_group.add_argument(
            "--qc-key",
            metavar="LABEL",
            type=str,
            default=None,
            help="Extra identifier for the QC session",
        )

    def parse_args(self, args: Sequence[str] | None = None) -> Namespace:
        """Parse CLI arguments."""
        return self.parser.parse_args(args)
