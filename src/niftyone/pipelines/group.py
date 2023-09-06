import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import fiftyone as fo
import pandas as pd
from bids2table import bids2table
from tqdm import tqdm

from niftyone import labels
from niftyone.typing import StrPath

IMG_EXTENSIONS = {".png", ".mp4"}
LABEL_CLS_LOOKUP = {"T1w": labels.MRIQCT1w, "bold": labels.MRIQCBold}


def group_pipeline(
    bids_dir: StrPath,
    out_dir: StrPath,
    ds_name: Optional[str] = None,
    overwrite: bool = False,
):
    """
    NiftyOne group pipeline. Collects image samples into a FiftyOne dataset and exports
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
        _add_group_samples(samples, group_index)

    logging.info("Collected samples: %d", len(samples))
    logging.info("Adding samples to the dataset")
    dataset.add_samples(samples)

    logging.info("Dataset group slices: %s", dataset.group_slices)
    logging.info("Dataset media types: %s", dataset.group_media_types)

    logging.info("Exporting dataset")
    dataset.export(
        export_dir=str(out_dir),
        dataset_type=fo.types.FiftyOneDataset,
        export_media=False,
        rel_dir=str(out_dir),
    )

    logging.info("Done! elapsed: %.2fs", time.monotonic() - tic)


def _add_group_samples(samples: List[fo.Sample], group_index: pd.DataFrame):
    group = fo.Group()

    qc_metrics = _load_qc_metrics(group_index)

    for _, record in group_index.iterrows():
        filepath = Path(record.file_path)

        if filepath.suffix in IMG_EXTENSIONS:
            # create sample
            element = f"{record.datatype}/{record.suffix}/{record.desc}"
            sample = fo.Sample(filepath=filepath, group=group.element(element))

            # add entity fields
            for k, v in record.items():
                if k not in {"file_path", "ext"} and not pd.isna(v):
                    sample[k] = v

            # add qc labels
            label_cls = LABEL_CLS_LOOKUP[record.suffix]
            sample[label_cls.__name__] = label_cls(**qc_metrics)

            samples.append(sample)


def _load_qc_metrics(group_index: pd.DataFrame) -> Dict[str, Any]:
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
