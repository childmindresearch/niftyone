import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache, partial
from pathlib import Path
from typing import List, Optional

import nibabel as nib
import numpy as np
import pandas as pd
from bids2table import bids2table
from bids2table.extractors.entities import BIDSEntities
from elbow.utils import cpu_count, setup_logging
from matplotlib import pyplot as plt

import niftyone.image as noimg
from niftyone.figures.bold import bold_mean_std, carpet_plot
from niftyone.figures.multi_view import slice_video, three_view_frame, three_view_video
from niftyone.typing import StrPath


def participant_raw(
    dataset_root: StrPath,
    out_root: StrPath,
    sub: Optional[str] = None,
    mriqc_root: Optional[StrPath] = None,
    workers: Optional[int] = None,
    overwrite: bool = False,
):
    """
    Participant-level niftyone raw MRI pipeline.
    """
    dataset_root = Path(dataset_root)
    out_root = Path(out_root)

    default_mriqc_root = dataset_root / "derivatives" / "mriqc"
    if mriqc_root is None and default_mriqc_root.exists():
        mriqc_root = default_mriqc_root
    else:
        mriqc_root = Path(mriqc_root)

    if workers is None:
        workers = 1
    elif workers == -1:
        workers = cpu_count()
    elif workers <= 0:
        raise ValueError(f"Invalid workers {workers}; expected -1 or > 0")

    logging.info(
        "Starting niftyone participant raw pipeline:"
        f"\n\tdataset: {dataset_root}"
        f"\n\tout: {out_root}"
        f"\n\tsubject: {sub}"
        f"\n\tmriqc: {mriqc_root}"
        f"\n\tworkers: {workers}"
        f"\n\toverwrite: {overwrite}"
    )

    logging.info("Loading dataset index")
    index = bids2table(dataset_root, workers=workers)

    if sub is None:
        subs = np.unique(index["entities"]["sub"])
        logging.info("Found %d subjects", len(subs))
    else:
        subs = [sub]

    _worker = partial(
        _participant_raw_worker,
        workers=workers,
        subs=subs,
        index=index,
        out_root=out_root,
        mriqc_root=mriqc_root,
        overwrite=overwrite,
        # propagate log level into worker processes
        log_level=logging.getLogger().level if workers > 1 else None,
    )

    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            futures_to_id = {pool.submit(_worker, ii): ii for ii in range(workers)}

            for future in as_completed(futures_to_id):
                try:
                    future.result()
                except Exception as exc:
                    worker_id = futures_to_id[future]
                    logging.warning(
                        "Generated exception in worker %d", worker_id, exc_info=exc
                    )
    else:
        _worker(0)


def _participant_raw_worker(
    worker_id: int,
    *,
    workers: int,
    subs: List[str],
    index: pd.DataFrame,
    out_root: Path,
    mriqc_root: Optional[Path] = None,
    overwrite: bool = False,
    log_level: Optional[int] = None,
):
    if log_level is not None:
        setup_logging(log_level)

    # find current worker's partition of subjects
    if workers > 1:
        subs = np.array_split(subs, workers)[worker_id]

    for sub in subs:
        _participant_raw_single(
            sub=sub,
            index=index,
            out_root=out_root,
            mriqc_root=mriqc_root,
            overwrite=overwrite,
        )


def _participant_raw_single(
    sub: str,
    index: pd.DataFrame,
    out_root: Path,
    mriqc_root: Optional[Path] = None,
    overwrite: bool = False,
):
    tic = time.monotonic()
    logging.info("Generating raw figures for subject: %s", sub)

    entities: pd.DataFrame = index["entities"]
    images = index.loc[
        (entities["sub"] == sub)
        & (entities["suffix"].isin({"T1w", "bold"}))
        & (entities["ext"].isin({".nii", ".nii.gz"}))
    ]
    if len(images) == 0:
        logging.info("Found no images")
        return

    logging.info(
        "Found %d images:\n\t%s",
        len(images),
        "\n\t".join(images["file"]["file_path"].tolist()),
    )

    for _, record in images.iterrows():
        if record["entities"]["suffix"] == "T1w":
            _participant_raw_t1w(
                record, out_root, mriqc_root=mriqc_root, overwrite=overwrite
            )
        elif record["entities"]["suffix"] == "bold":
            _participant_raw_bold(
                record, out_root, mriqc_root=mriqc_root, overwrite=overwrite
            )

    logging.info(
        "Done processing subject: %s; elapsed: %.2fs", sub, time.monotonic() - tic
    )


def _participant_raw_t1w(
    record: pd.Series,
    out_root: Path,
    mriqc_root: Optional[Path] = None,
    overwrite: bool = False,
):
    entities = BIDSEntities.from_dict(record["entities"])

    img_path = Path(record["file"]["file_path"])
    logging.info("Processing: %s", img_path)
    img = nib.load(img_path)
    img = noimg.to_iso_ras(img)

    out_path = entities.with_update(desc="threeView", ext=".png").to_path(
        prefix=out_root
    )
    out_path.parent.mkdir(exist_ok=True, parents=True)
    if not out_path.exists() or overwrite:
        logging.info("Generating: %s", out_path)
        three_view_frame(img, out=out_path)

    out_path = entities.with_update(desc="sliceVideo", ext=".mp4").to_path(
        prefix=out_root
    )
    if not out_path.exists() or overwrite:
        logging.info("Generating: %s", out_path)
        slice_video(img, out=out_path)

    if mriqc_root is not None and mriqc_root.exists():
        _mriqc_metrics_tsv(record, entities, out_root, mriqc_root, overwrite=overwrite)


def _participant_raw_bold(
    record: pd.Series,
    out_root: Path,
    mriqc_root: Optional[Path],
    overwrite: bool = False,
):
    entities = BIDSEntities.from_dict(record["entities"])

    img_path = Path(record["file"]["file_path"])
    logging.info("Processing: %s", img_path)
    img = nib.load(img_path)
    img = noimg.to_iso_ras(img)

    out_path = entities.with_update(desc="threeView", ext=".png").to_path(
        prefix=out_root
    )
    out_path.parent.mkdir(exist_ok=True, parents=True)
    if not out_path.exists() or overwrite:
        logging.info("Generating: %s", out_path)
        three_view_frame(img, out=out_path)

    out_path = entities.with_update(desc="threeViewVideo", ext=".mp4").to_path(
        prefix=out_root
    )
    if not out_path.exists() or overwrite:
        logging.info("Generating: %s", out_path)
        three_view_video(img, out=out_path)

    out_path = entities.with_update(desc="carpet", ext=".png").to_path(prefix=out_root)
    if not out_path.exists() or overwrite:
        logging.info("Generating: %s", out_path)
        fig = carpet_plot(img, out=out_path)
        plt.close(fig)

    out_path = entities.with_update(desc="meanStd", ext=".png").to_path(prefix=out_root)
    if not out_path.exists() or overwrite:
        logging.info("Generating: %s", out_path)
        bold_mean_std(img, out=out_path)

    if mriqc_root is not None and mriqc_root.exists():
        _mriqc_metrics_tsv(record, entities, out_root, mriqc_root, overwrite=overwrite)


def _mriqc_metrics_tsv(
    record: pd.Series,
    entities: BIDSEntities,
    out_root: Path,
    mriqc_root: Path,
    overwrite: bool = False,
):
    out_path = entities.with_update(desc="QCMetrics", ext=".tsv").to_path(
        prefix=out_root
    )
    if out_path.exists() and not overwrite:
        return

    metrics = _load_mriqc_group_metrics(mriqc_root, entities.suffix)

    if metrics is not None:
        # find metrics matching this image
        query = record["entities"].dropna().to_dict()
        for k in ["datatype", "ext", "extra_entities"]:
            query.pop(k, None)
        query = " and ".join(f"{k} == {repr(v)}" for k, v in query.items())
        img_metrics = metrics.query(query)

        if len(img_metrics) > 0:
            logging.info("Generating: %s", out_path)
            img_metrics.to_csv(out_path, sep="\t", index=False)


@lru_cache(maxsize=2)
def _load_mriqc_group_metrics(
    mriqc_root: Path, suffix: str = "T1w"
) -> Optional[pd.DataFrame]:
    # TODO: this should probably be factored somewhere else

    metrics_path = mriqc_root / f"group_{suffix}.tsv"
    if not metrics_path.exists():
        return None

    metrics = pd.read_csv(metrics_path, sep="\t")

    # parse bids names to entities
    bids_names = metrics["bids_name"]
    entities = [BIDSEntities.from_path(bids_name).to_dict() for bids_name in bids_names]
    entities = pd.DataFrame.from_records(entities)

    # drop all NA columns
    entities.dropna(axis=1, how="all", inplace=True)

    # cat and set entities to index
    metrics = pd.concat([entities, metrics], axis=1)
    metrics.set_index(entities.columns.to_list(), inplace=True)
    return metrics
