"""Functions for participant-level metrics."""

import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd
from bids2table.entities import BIDSEntities


def gen_niftyone_metrics_tsv(
    record: pd.Series,
    entities: BIDSEntities,
    out_dir: Path,
    qc_dir: Path,
    overwrite: bool = False,
) -> None:
    """Filter for metrics associated with image."""
    out_path = entities.with_update(desc="QCMetrics", ext=".tsv").to_path(
        prefix=out_dir
    )
    if out_path.exists() and not overwrite:
        return

    metrics = _load_qc_group_metrics(qc_dir, entities.suffix)
    if metrics is None:
        return

    # Find metrics matching image
    query = {
        k: v
        for k, v in record["ent"].dropna().to_dict().items()
        if k not in ["datatype", "ext", "extra_entities"]
    }
    query_str = " and ".join(f"{k} == {repr(v)}" for k, v in query.items())
    img_metrics = metrics.query(query_str)

    if len(img_metrics) > 0:
        logging.info("Generating: %s", out_path)
        img_metrics.to_csv(out_path, sep="\t", index=False)


@lru_cache(maxsize=2)
def _load_qc_group_metrics(qc_dir: Path, suffix: str = "T1w") -> pd.DataFrame | None:
    metrics_path = qc_dir / f"group_{suffix}.tsv"
    if not metrics_path.exists():
        return None

    metrics = pd.read_csv(metrics_path, sep="\t")

    # Parse bids names to entities, dropping NA columns
    entities = pd.DataFrame.from_records(
        [
            BIDSEntities.from_path(bids_name).to_dict()
            for bids_name in metrics["bids_name"]
        ]
    )
    entities.dropna(axis=1, how="all", inplace=True)

    # Concatenate entities, setting them as index
    metrics = pd.concat([entities, metrics], axis=1)
    metrics.set_index(entities.columns.to_list(), inplace=True)

    return metrics
