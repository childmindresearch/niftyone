"""Entrypoint of application."""

import argparse
import json
from pathlib import Path

from elbow.utils import setup_logging

import niftyone
from niftyone import pipelines


def _make_dataset_description(out_dir: Path) -> None:
    description = {
        "Name": "NiftyOne",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "NiftyOne",
                "Version": f"{niftyone.__version__}",
                "CodeURL": "https://github.com/childmindresearch/niftyone",
            }
        ],
        "HowToAcknowledge": "Please cite our repo (https://github.com/childmindresearch/niftyone).",
        "License": "LGPL-2.1",
    }

    with (out_dir / "dataset_description.json").open("w") as f:
        json.dump(description, f, indent=4)


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="niftyone",
        usage="%(prog)s bids_dir output_dir analysis_level [options]",
        description="Help menu.",
    )

    # Common arguments
    parser.add_argument(
        "bids_dir",
        metavar="bids_dir",
        type=Path,
        help="Path to BIDS dataset",
    )
    parser.add_argument(
        "out_dir",
        metavar="output_dir",
        type=Path,
        help="Path to output directory",
    )
    parser.add_argument(
        "analysis_level",
        metavar="analysis_level",
        type=str,
        choices=["participant", "group", "launch"],
        help="Analysis level",
    )
    parser.add_argument(
        "--overwrite",
        "-x",
        help="Overwrite previous results",
        action="store_true",
    )
    parser.add_argument("--verbose", "-v", help="Verbose logging.", action="store_true")

    # Participant level options
    participant_group = parser.add_argument_group("Participant level options")
    participant_group.add_argument(
        "--participant-label",
        "--sub",
        metavar="LABEL",
        type=str,
        default=None,
        help="Participant to analyze (default: all)",
    )
    participant_group.add_argument(
        "--index",
        metavar="PATH",
        type=Path,
        default=None,
        help="Path to pre-computed bids2table index (default: {bids_dir}/index.b2t)",
    )
    participant_group.add_argument(
        "--qc-dir",
        metavar="PATH",
        type=Path,
        default=None,
        help=(
            "Path to pre-computed QC outputs " "(default: {bids_dir}/derivatives/mriqc)"
        ),
    )
    participant_group.add_argument(
        "--workers",
        "-w",
        metavar="COUNT",
        type=int,
        help="Number of worker processes. Setting to -1 runs as many processes as "
        "there are cores available. (default: 1)",
        default=1,
    )

    # Group level arguments
    group_group = parser.add_argument_group("Group level options")
    group_group.add_argument(
        "--ds-name",
        metavar="DATASET",
        type=str,
        default=None,
        help="Name of FiftyOne dataset.",
    )

    # Launch level arguments
    launch_group = parser.add_argument_group("Launch level options")
    launch_group.add_argument(
        "--qc-key",
        metavar="LABEL",
        type=str,
        default=None,
        help="Extra identifier for the QC session",
    )

    return parser


def main() -> None:
    """NiftyOne BIDS-app entrypoint."""
    parser = _create_parser()
    args = parser.parse_args()

    setup_logging("INFO" if args.verbose else "ERROR")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    _make_dataset_description(out_dir=out_dir)

    match args.analysis_level:
        case "participant":
            pipelines.participant(
                bids_dir=args.bids_dir,
                out_dir=out_dir,
                sub=args.participant_label,
                index_path=args.index,
                qc_dir=args.qc_dir,
                workers=args.workers,
                overwrite=args.overwrite,
                verbose=args.verbose,
            )
        case "group":
            pipelines.group(
                bids_dir=args.bids_dir,
                out_dir=out_dir,
                ds_name=args.ds_name,
                overwrite=args.overwrite,
            )
        case "launch":
            niftyone.launch(bids_dir=args.bids_dir, out_dir=out_dir, qc_key=args.qc_key)
        case _:
            raise NotImplementedError(
                f"Analysis level {args.analysis_level} not implemented."
            )


if __name__ == "__main__":
    main()
