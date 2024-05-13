"""Raw participant-label pipeline."""

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from bids2table import BIDSTable, bids2table
from bids2table.entities import BIDSEntities
from elbow.utils import cpu_count, setup_logging
from matplotlib import pyplot as plt

import niclips.image as noimg
from niclips.figures.bold import bold_mean_std, carpet_plot
from niclips.figures.multi_view import slice_video, three_view_frame, three_view_video
from niftyone import metrics, typing


def participant_raw_pipeline(
    bids_dir: typing.StrPath,
    out_dir: typing.StrPath,
    sub: str | None = None,
    index_path: typing.StrPath | None = None,
    qc_dir: typing.StrPath | None = None,
    workers: int = 1,
    overwrite: bool = False,
    verbose: bool = False,
) -> None:
    """Participant-level niftyone raw MRI pipeline."""
    bids_dir = Path(bids_dir)
    out_dir = Path(out_dir)

    default_qc_dir = bids_dir / "derivatives" / "mriqc"
    if qc_dir is None and default_qc_dir.exists():
        qc_dir = default_qc_dir
    elif qc_dir is not None:
        qc_dir = Path(qc_dir)

    if workers == -1:
        workers = cpu_count()
    elif workers <= 0:
        raise ValueError(f"Invalid workers {workers}; expected -1 or > 0")

    setup_logging("INFO" if verbose else "WARNING", max_repeats=None)
    logging.info(
        "Starting niftyone participant raw pipeline:"
        f"\n\tdataset: {bids_dir}"
        f"\n\tout: {out_dir}"
        f"\n\tsubject: {sub}"
        f"\n\tindex: {index_path}"
        f"\n\tqc: {qc_dir}"
        f"\n\tworkers: {workers}"
        f"\n\toverwrite: {overwrite}"
    )

    logging.info("Loading dataset index")
    index = bids2table(bids_dir, index_path=index_path, workers=workers)

    if sub is None:
        subs = sorted(index.subjects)
        logging.info("Found %d subjects", len(subs))
    else:
        subs = [sub]

    _worker = partial(
        _participant_raw_worker,
        workers=workers,
        subs=subs,
        index=index,
        out_dir=out_dir,
        qc_dir=qc_dir,
        overwrite=overwrite,
        verbose=verbose,
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
    subs: list[str],
    index: BIDSTable,
    out_dir: Path,
    qc_dir: Path | None = None,
    overwrite: bool = False,
    verbose: bool = False,
) -> None:
    # reset logger for each worker
    # TODO: this is a hack, should be fixed in elbow
    setup_logging("INFO" if verbose else "WARNING", max_repeats=None)

    # find current worker's partition of subjects
    if workers > 1:
        subs = np.array_split(subs, workers)[worker_id]  # type: ignore [assignment]

    for sub in subs:
        _participant_raw_single(
            sub=sub,
            index=index,
            out_dir=out_dir,
            qc_dir=qc_dir,
            overwrite=overwrite,
        )


def _participant_raw_single(
    sub: str,
    index: BIDSTable,
    out_dir: Path,
    qc_dir: Path | None = None,
    overwrite: bool = False,
) -> None:
    tic = time.monotonic()
    logging.info("Generating raw figures for subject: %s", sub)

    images = (
        index.filter("sub", sub)
        .filter("suffix", items={"T1w", "bold"})
        .filter("ext", items={".nii", ".nii.gz"})
    )
    if len(images) == 0:
        logging.info("Found no images")
        return

    logging.info(
        "Found %d images:\n\t%s",
        len(images),
        "\n\t".join(images.finfo["file_path"].tolist()),
    )

    for _, record in images.nested.iterrows():
        if record["ent"]["suffix"] == "T1w":
            _participant_raw_t1w(record, out_dir, qc_dir=qc_dir, overwrite=overwrite)
        elif record["ent"]["suffix"] == "bold":
            _participant_raw_bold(record, out_dir, qc_dir=qc_dir, overwrite=overwrite)

    logging.info(
        "Done processing subject: %s; elapsed: %.2fs", sub, time.monotonic() - tic
    )


def _participant_raw_t1w(
    record: pd.Series,
    out_dir: Path,
    qc_dir: Path | None = None,
    overwrite: bool = False,
) -> None:
    entities = BIDSEntities.from_dict(record["ent"])

    img_path = Path(record["finfo"]["file_path"])
    logging.info("Processing: %s", img_path)
    img = nib.nifti1.load(img_path)
    img = noimg.to_iso_ras(img)

    out_path = entities.with_update(desc="threeView", ext=".png").to_path(
        prefix=out_dir
    )
    out_path.parent.mkdir(exist_ok=True, parents=True)
    if not out_path.exists() or overwrite:
        logging.info("Generating: %s", out_path)
        three_view_frame(img, out=out_path)

    out_path = entities.with_update(desc="sliceVideo", ext=".mp4").to_path(
        prefix=out_dir
    )
    if not out_path.exists() or overwrite:
        logging.info("Generating: %s", out_path)
        slice_video(img, out=out_path)

    if qc_dir is not None and qc_dir.exists():
        metrics.gen_niftyone_metrics_tsv(
            record, entities, out_dir, qc_dir, overwrite=overwrite
        )


def _participant_raw_bold(
    record: pd.Series,
    out_dir: Path,
    qc_dir: Path | None,
    overwrite: bool = False,
) -> None:
    entities = BIDSEntities.from_dict(record["ent"])

    img_path = Path(record["finfo"]["file_path"])
    logging.info("Processing: %s", img_path)
    img = nib.nifti1.load(img_path)
    img = noimg.to_iso_ras(img)

    out_path = entities.with_update(desc="threeViewVideo", ext=".mp4").to_path(
        prefix=out_dir
    )
    out_path.parent.mkdir(exist_ok=True, parents=True)
    if not out_path.exists() or overwrite:
        logging.info("Generating: %s", out_path)
        three_view_video(img, out=out_path)

    out_path = entities.with_update(desc="carpet", ext=".png").to_path(prefix=out_dir)
    if not out_path.exists() or overwrite:
        logging.info("Generating: %s", out_path)
        fig = carpet_plot(img, out=out_path)
        plt.close(fig)

    out_path = entities.with_update(desc="meanStd", ext=".png").to_path(prefix=out_dir)
    if not out_path.exists() or overwrite:
        logging.info("Generating: %s", out_path)
        bold_mean_std(img, out=out_path)

    if qc_dir is not None and qc_dir.exists():
        metrics.gen_niftyone_metrics_tsv(
            record, entities, out_dir, qc_dir, overwrite=overwrite
        )
