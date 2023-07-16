import argparse
import sys
from pathlib import Path

from elbow.utils import setup_logging

from niftyone.pipelines.participant_raw import participant_raw


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
        choices=["participant", "group"],
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
        "--nprocs",
        "-J",
        metavar="COUNT",
        type=int,
        help=(
            "Number of worker processes. Setting to -1 runs as many procs as "
            "there are cpus. (default: 1)"
        ),
        default=1,
    )
    parser.add_argument("--verbose", "-v", help="Verbose logging.", action="store_true")

    args = parser.parse_args()

    setup_logging("INFO" if args.verbose else "ERROR")

    if args.analysis_level == "participant":
        participant_raw(
            bids_dir=args.bids_dir,
            out_dir=args.out_dir,
            sub=args.participant_label,
            index_path=args.index,
            mriqc_dir=args.mriqc_dir,
            nprocs=args.nprocs,
        )
    else:
        raise NotImplementedError(
            f"Analysis level {args.analysis_level} not implemented"
        )


if __name__ == "__main__":
    sys.exit(main())
