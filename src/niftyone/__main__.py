"""Entrypoint of application."""

import json
from pathlib import Path

from elbow.utils import setup_logging

import niftyone
from niftyone import pipelines
from niftyone.cli import NiftyOneArgumentParser


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


def main() -> None:
    """NiftyOne BIDS-app entrypoint."""
    parser = NiftyOneArgumentParser()
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
