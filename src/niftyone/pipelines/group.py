"""Pipeline for group-level."""

import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import fiftyone as fo
import pandas as pd
from bids2table import bids2table
from bids2table.entities import BIDSEntities
from PIL import Image
from tqdm import tqdm

from niclips.io import VideoWriter
from niftyone.metadata.tags import TAGS
from niftyone.typing import StrPath

IMG_EXTENSIONS = {".png", ".mp4"}


def group_pipeline(
    bids_dir: StrPath,
    out_dir: StrPath,
    ds_name: str | None = None,
    overwrite: bool = False,
) -> None:
    """NiftyOne group pipeline.

    Collects image samples into a FiftyOne dataset and exports
    to the output directory.
    """
    tic = time.monotonic()
    bids_dir = Path(bids_dir)
    out_dir = Path(out_dir)
    if ds_name is None:
        ds_name = Path(bids_dir).name

    logging.info(
        "Starting niftyone group pipeline:"
        f"\n\tdataset: {bids_dir}"
        f"\n\tout: {out_dir}"
        f"\n\tds_name: {ds_name}"
        f"\n\toverwrite: {overwrite}"
    )

    logging.info("Loading dataset index")
    index = bids2table(out_dir, persistent=True, overwrite=True)
    if len(index) == 0:
        logging.warning("No files found in output dir %s", out_dir)
        return

    # extract out just the entities and the file paths
    entities = index.ent.copy()
    entities.dropna(axis=1, how="all", inplace=True)
    entities.drop("extra_entities", axis=1, inplace=True)
    paths = index.finfo[["file_path"]]
    index = pd.concat([entities, paths], axis=1)

    by = [k for k in entities.columns if k not in {"desc", "ext"}]
    grouped = index.groupby(by, dropna=False)

    logging.info("Collecting dataset samples for %d groups", len(grouped))

    dataset: fo.Dataset = fo.Dataset(ds_name, persistent=True, overwrite=overwrite)

    dataset.add_group_field("group")
    samples = []
    for _, group_index in tqdm(grouped):
        group_samples = _get_group_samples(group_index)
        samples.extend(group_samples)

    logging.info("Collected samples: %d", len(samples))
    logging.info("Adding samples to the dataset")
    dataset.add_samples(samples, dynamic=True)

    logging.info("Dataset group slices: %s", dataset.group_slices)
    logging.info("Dataset media types: %s", dataset.group_media_types)

    # Dummy samples for initializing tags. Currently creating empty tags is not
    # supported in FiftyOne, this is a workaround.
    # TODO: update if/when FiftyOne supports empty tags
    dummy_group = fo.Group()
    dummy_samples = []
    for element, modality in dataset.group_media_types.items():
        sample = _get_dummy_sample(dummy_group, element, modality, out_dir)
        dummy_samples.append(sample)
    dataset.add_samples(dummy_samples)

    logging.info("Exporting dataset")
    fo_dir = out_dir / "fiftyone"
    dataset.export(
        export_dir=str(fo_dir),
        dataset_type=fo.types.FiftyOneDataset,
        export_media=False,
        rel_dir=str(fo_dir),
        overwrite=overwrite,
    )

    logging.info("Done! elapsed: %.2fs", time.monotonic() - tic)


def _get_group_samples(group_index: pd.DataFrame) -> list[fo.Sample]:
    samples = []
    group = fo.Group()

    # Grab QC label (and sub-labels) + metrics
    qc_suffix = f"QC{group_index.iloc[0].suffix.capitalize()}"
    qc_metrics = _load_qc_metrics(group_index)
    qc_vars = fo.DynamicEmbeddedDocument(**qc_metrics)

    for _, record in group_index.iterrows():
        filepath = Path(record.file_path)

        if filepath.suffix in IMG_EXTENSIONS:
            # create sample
            element = f"{record.datatype}/{record.suffix}/{record.desc}"
            group_key = _get_group_key(record)

            sample = fo.Sample(
                filepath=filepath,
                group=group.element(element),
                group_key=_get_group_label(group_key),
                **{qc_suffix: qc_vars},
            )

            # add entity fields
            for k, v in record.items():
                if k not in {"file_path", "ext"} and not pd.isna(v):
                    sample[k] = v

            samples.append(sample)
    return samples


def _load_qc_metrics(group_index: pd.DataFrame) -> dict[str, Any]:
    # Find the metrics record
    metrics = group_index.query("desc == 'QCMetrics' and ext == '.tsv'")
    assert len(metrics) in {0, 1}, "Expected at most QCMetrics tsv"

    if len(metrics) == 0:
        return {}

    # Read the metrics from the tsv
    metrics = pd.read_csv(metrics.iloc[0]["file_path"], sep="\t")
    metrics.drop("bids_name", axis=1, inplace=True)
    metrics = metrics.iloc[0].to_dict()
    return metrics


def _get_group_key(record: pd.Series) -> fo.Classification:
    group_ent = BIDSEntities.from_dict(
        {k: v for k, v in record.items() if k not in {"desc", "ext", "file_path"}}
    )
    key = Path(group_ent.to_path(valid_only=True)).name
    return key


@lru_cache
def _get_group_label(key: str) -> fo.Classification:
    # Cache the labels so that tags are shared
    label = fo.Classification(label=key)
    return label


def _get_dummy_sample(
    group: fo.Group, element: str, modality: str, out_dir: Path
) -> fo.Sample:
    dummy_dir = out_dir / ".DUMMY"
    dummy_dir.mkdir(exist_ok=True)

    name = element.replace("/", "_")
    if modality == "image":
        dummy_path = dummy_dir / (name + ".png")
        _dummy_image(dummy_path)
    elif modality == "video":
        dummy_path = dummy_dir / (name + ".mp4")
        _dummy_video(dummy_path)
    else:
        raise ValueError(f"Unknown modality: {modality}")

    sample = fo.Sample(dummy_path, group=group.element(element))
    label = fo.Classification(label="DUMMY")
    label.tags = TAGS
    sample["group_key"] = label
    return sample


def _dummy_image(path: StrPath) -> None:
    Image.new("RGB", (256, 256), (0, 0, 0)).save(path)


def _dummy_video(path: StrPath) -> None:
    with VideoWriter(path, 10) as writer:
        for _ in range(10):
            writer.put(Image.new("RGB", (256, 256), (0, 0, 0)))
