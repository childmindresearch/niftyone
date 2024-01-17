import argparse
import json
import sys
from pathlib import Path

from elbow.utils import setup_logging

import niftyone
from niftyone.launch import launch
from niftyone.pipelines.group import group_pipeline
from niftyone.pipelines.participant_raw import participant_raw_pipeline


def _make_dataset_description():
    description = {
        "Name": "NiftyOne",
        "BIDSVersion": "1.8.0",
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
    return description


def main():
    """
    NiftyOne BIDS-app entrypoint.
    """
    parser = argparse.ArgumentParser("niftyone")

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
        "--participant-label",
        "--sub",
        metavar="LABEL",
        type=str,
        default=None,
        help="Participant to analyze (default: all)",
    )
    parser.add_argument(
        "--index",
        metavar="PATH",
        type=Path,
        default=None,
        help="Path to pre-computed bids2table index (default: {bids_dir}/index.b2t)",
    )
    parser.add_argument(
        "--mriqc-dir",
        metavar="PATH",
        type=Path,
        default=None,
        help=(
            "Path to pre-computed MRIQC outputs "
            "(default: {bids_dir}/derivatives/mriqc)"
        ),
    )
    parser.add_argument(
        "--qc-key",
        metavar="LABEL",
        type=str,
        default=None,
        help="Extra identifier for the QC session",
    )
    parser.add_argument(
        "--workers",
        "-w",
        metavar="COUNT",
        type=int,
        help="Number of worker processes. Setting to -1 runs as many processes as "
        "there are cores available. (default: 1)",
        default=1,
    )
    parser.add_argument(
        "--overwrite",
        "-x",
        help="Overwrite previous results",
        action="store_true",
    )
    parser.add_argument("--verbose", "-v", help="Verbose logging.", action="store_true")

    args = parser.parse_args()

    setup_logging("INFO" if args.verbose else "ERROR")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    description = _make_dataset_description()
    with (out_dir / "dataset_description.json").open("w") as f:
        json.dump(description, f, indent=4)

    if args.analysis_level == "participant":
        participant_raw_pipeline(
            bids_dir=args.bids_dir,
            out_dir=args.out_dir,
            sub=args.participant_label,
            index_path=args.index,
            mriqc_dir=args.mriqc_dir,
            workers=args.workers,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
    elif args.analysis_level == "group":
        group_pipeline(
            bids_dir=args.bids_dir, out_dir=args.out_dir, overwrite=args.overwrite
        )

    elif args.analysis_level == "launch":
        launch(bids_dir=args.bids_dir, out_dir=args.out_dir, qc_key=args.qc_key)

    else:
        raise NotImplementedError(
            f"Analysis level {args.analysis_level} not implemented"
        )


if __name__ == "__main__":
    sys.exit(main())
