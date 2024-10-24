"""Launch analysis-level."""

import logging
from pathlib import Path

import fiftyone as fo

from niftyone.metadata.tags import GroupTags
from niftyone.typing import StrPath

# Disable fiftyone tracking
fo.config.do_not_track = True


def launch(
    bids_dir: StrPath,
    out_dir: StrPath,
    ds_name: str | None = None,
    qc_key: str | None = None,
) -> None:
    """Launch FiftyOne application to visualize a dataset."""
    bids_dir = Path(bids_dir)
    out_dir = Path(out_dir)
    if ds_name is None:
        ds_name = Path(bids_dir).name
    if qc_key:
        ds_name = f"{ds_name}-{qc_key}"

    if fo.dataset_exists(ds_name):
        logging.info("Loading dataset from FiftyOne database")
        dataset = fo.load_dataset(ds_name)
    else:
        logging.info("Loading dataset from %s", out_dir / "fiftyone")
        try:
            dataset = fo.Dataset.from_dir(
                dataset_dir=out_dir / "fiftyone",
                dataset_type=fo.types.FiftyOneDataset,
                name=ds_name,
                persistent=True,
            )
        except FileNotFoundError as err:
            raise FileNotFoundError(
                f"FiftyOne dataset not found in {out_dir}. "
                "Did you run the participant and group level?"
            ) from err

    tags_path = out_dir / "QC" / f"{ds_name}_tags.json"
    if tags_path.exists():
        logging.info("Loading QC tags from %s", tags_path)
        tags = GroupTags.from_json(tags_path)
        tags.apply(dataset)

    session = fo.launch_app(dataset)
    try:
        print("Press ctrl+c to exit...")
        session.wait()
    except KeyboardInterrupt:
        pass
    finally:
        logging.info("Saving QC tags to %s", tags_path)
        group_tags = GroupTags.from_dataset(dataset)
        tags_path.parent.mkdir(exist_ok=True)
        group_tags.to_json(tags_path)
        session.close()
