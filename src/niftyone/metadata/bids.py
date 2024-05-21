"""Functions associated with BIDS dataset metadata."""

import json
from pathlib import Path

import niftyone


def make_dataset_description(out_dir: Path) -> None:
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

    with (out_dir / "dataset_description.json").open("w") as f:
        json.dump(description, f, indent=4)
