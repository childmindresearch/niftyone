"""Utilties for handling BIDS dataset associated metadata."""

import json
from pathlib import Path

import niftyone


def make_dataset_description(out_dir: Path, overwrite: bool) -> None:
    """Create dataset_description.json for BIDS dataset."""
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

    ds_fpath = out_dir.joinpath("dataset_description.json")
    if not ds_fpath.exists() or overwrite:
        with ds_fpath.open("w") as f:
            json.dump(description, f, indent=4)
