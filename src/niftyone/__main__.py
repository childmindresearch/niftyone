"""Entrypoint of application."""

from pathlib import Path

from elbow.utils import setup_logging

import niftyone
from niftyone import pipelines
from niftyone.cli import NiftyOneArgumentParser
from niftyone.metadata import bids


def main() -> None:
    """NiftyOne BIDS-app entrypoint."""
    parser = NiftyOneArgumentParser()
    args = parser.parse_args()

    setup_logging("INFO" if args.verbose else "ERROR")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    bids.make_dataset_description(out_dir=out_dir)

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
