from pathlib import Path
from typing import Optional

import fiftyone as fo

from niftyone.typing import StrPath


def launch(bids_dir: StrPath, out_dir: StrPath, ds_name: Optional[str] = None):
    """
    Launch the FiftyOne app to visualize a dataset (after it has been generated).
    """
    bids_dir = Path(bids_dir)
    out_dir = Path(out_dir)
    if ds_name is None:
        ds_name = Path(bids_dir).name

    dataset = fo.Dataset.from_dir(
        dataset_dir=out_dir,
        dataset_type=fo.types.FiftyOneDataset,
        name=ds_name,
    )

    session = fo.launch_app(dataset)
    session.wait()
