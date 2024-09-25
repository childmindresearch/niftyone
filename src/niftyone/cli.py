"""CLI-related functions."""

from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from collections.abc import Sequence
from pathlib import Path


class NiftyOneArgumentParser:
    """NiftyOne CLI parser."""

    def __init__(self) -> None:
        self.parser = ArgumentParser(
            prog="niftyone",
            usage="%(prog)s bids_dir output_dir analysis_level [options]",
            formatter_class=RawDescriptionHelpFormatter,
            description="""
NiftyOne is a comphrensive tool designed to aid large-scale
QC of BIDS datasets through visualization and quantitative
metrics.

The different analysis levels perform the following tasks:
    * participant - generation of figures
    * group - collects figures + qc metrics as NiftyOne samples
    * launch - launches NiftyOne application""",
        )
        self._add_common_args()
        self._add_participant_args()
        self._add_group_launch_args()

    def _add_common_args(self) -> None:
        """Common (non-analysis specific) arguments."""
        self.parser.add_argument(
            "bids_dir",
            metavar="bids_dir",
            type=Path,
            help="path to BIDS dataset",
        )
        self.parser.add_argument(
            "out_dir",
            metavar="output_dir",
            type=Path,
            help="path to output directory",
        )
        self.parser.add_argument(
            "analysis_level",
            metavar="analysis_level",
            type=str,
            choices=["participant", "group", "launch"],
            help="analysis level - one of [%(choices)s]",
        )
        self.parser.add_argument(
            "--overwrite",
            "-x",
            help="overwrite previous results",
            action="store_true",
        )
        self.parser.add_argument(
            "--verbose", "-v", help="verbose logging.", action="store_true"
        )

    def _add_participant_args(self) -> None:
        """Participant-level CLI arguments."""
        self.participant_level = self.parser.add_argument_group(
            title="participant level options",
            description="Generates figures for individual participants.",
        )
        self.participant_level.add_argument(
            "--participant-label",
            "--sub",
            metavar="LABEL",
            type=str,
            default=None,
            help="participant to analyze.",
        )
        self.participant_level.add_argument(
            "--index",
            metavar="PATH",
            type=Path,
            default=None,
            help="bids2table index path",
        )
        self.participant_level.add_argument(
            "--qc-dir",
            metavar="PATH",
            type=Path,
            default=None,
            help="pre-computed QC metrics if available",
        )
        self.participant_level.add_argument(
            "--plugin-dir",
            metavar="PATH",
            type=Path,
            default=None,
            help="directory to search for plugins in;plugins should be "
            "prepended with 'niftyone_'",
        )
        self.participant_level.add_argument(
            "--config",
            metavar="PATH",
            type=Path,
            default=None,
            help=(
                "filters to apply to bids2table for figure generation - "
                "if none provided, create all available figures"
            ),
        )
        self.participant_level.add_argument(
            "--workers",
            "-w",
            metavar="COUNT",
            type=int,
            help="number of worker processes - setting to -1 uses all available cores "
            "(default: %(default)d)",
            default=1,
        )

    def _add_group_launch_args(self) -> None:
        """Application group / launch CLI arguments."""
        self.launch_group = self.parser.add_argument_group(
            title="group / launch level options",
        )
        self.launch_group.add_argument(
            "--ds-name",
            metavar="DATASET",
            type=str,
            default=None,
            help="name of NiftyOne dataset (default: bids_dir)",
        )
        self.launch_group.add_argument(
            "--qc-key",
            metavar="LABEL",
            type=str,
            default=None,
            help="extra identifier for the QC session (default: %(default)s)",
        )

    def parse_args(self, args: Sequence[str] | None = None) -> Namespace:
        """Parse CLI arguments."""
        return self.parser.parse_args(args)
