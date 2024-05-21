import json
from pathlib import Path

import niftyone
from niftyone.metadata import bids


def test_make_dataset_description(tmp_path: Path) -> None:
    out_dir = tmp_path
    bids.make_dataset_description(out_dir=out_dir)

    # Check file created
    json_fpath = out_dir / "dataset_description.json"
    assert json_fpath.exists()

    # Read & validate content
    with json_fpath.open() as f:
        description = json.load(f)
    assert description["Name"] == "NiftyOne"
    assert description["BIDSVersion"] == "1.9.0"
    assert description["DatasetType"] == "derivative"
    assert description["GeneratedBy"][0]["Name"] == "NiftyOne"
    assert description["GeneratedBy"][0]["Version"] == niftyone.__version__
    assert (
        description["GeneratedBy"][0]["CodeURL"]
        == "https://github.com/childmindresearch/niftyone"
    )
    assert description["License"] == "LGPL-2.1"
