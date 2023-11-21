import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache, partial
from pathlib import Path
from typing import List, Optional

import nibabel as nib
import numpy as np
import pandas as pd
from bids2table import BIDSTable, bids2table
from bids2table.entities import BIDSEntities
from elbow.utils import cpu_count, setup_logging
from matplotlib import pyplot as plt

import niftyone.image as noimg
from niftyone.figures.bold import bold_mean_std, carpet_plot
from niftyone.figures.multi_view import slice_video, three_view_frame, three_view_video
from niftyone.typing import StrPath


def participant_brainmask_pipeline():
    bids_dir: StrPath,
    out_dir: StrPath,
    sub: Optional[str] = None,
    index_path: Optional[StrPath] = None,
    mriqc_dir: Optional[StrPath] = None,
    workers: Optional[int] = None,
    overwrite: bool = False,
    verbose: bool = False,
):
    """
    Participant-level niftyone brainmask MRI pipeline.
    """
    bids_dir = Path(bids_dir)
    out_dir = Path(out_dir)

    default_mriqc_dir = bids_dir / "derivatives" / "mriqc"
    if mriqc_dir is None and default_mriqc_dir.exists():
        mriqc_dir = default_mriqc_dir
    elif mriqc_dir is not None:
        mriqc_dir = Path(mriqc_dir)

    if workers is None:
        workers = 1
    elif workers == -1:
        workers = cpu_count()
    elif workers <= 0:
        raise ValueError(f"Invalid workers {workers}; expected -1 or > 0")

    setup_logging("INFO" if verbose else "WARNING", max_repeats=None)
    logging.info(
        "Starting niftyone participant brainmask pipeline:"
        f"\n\tdataset: {bids_dir}"
        f"\n\tout: {out_dir}"
        f"\n\tsubject: {sub}"
        f"\n\tindex: {index_path}"
        f"\n\tmriqc: {mriqc_dir}"
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
        _participant_brainmask_worker,
        workers=workers,
        subs=subs,
        index=index,
        out_dir=out_dir,
        mriqc_dir=mriqc_dir,
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

def _participant_brainmask_worker(
    worker_id: int,
    *,
    workers: int,
    subs: List[str],
    index: BIDSTable,
    out_dir: Path,
    mriqc_dir: Optional[Path] = None,
    overwrite: bool = False,
    verbose: bool = False,
):
    # reset logger for each worker
    # TODO: this is a hack, should be fixed in elbow
    setup_logging("INFO" if verbose else "WARNING", max_repeats=None)

    # find current worker's partition of subjects
    if workers > 1:
        subs = np.array_split(subs, workers)[worker_id]

    for sub in subs:
        _participant_brainmask_single(
            sub=sub,
            index=index,
            out_dir=out_dir,
            mriqc_dir=mriqc_dir,
            overwrite=overwrite,
        )

def _participant_brainmask_single(
    sub: str,
    index: BIDSTable,
    out_dir: Path,
    mriqc_dir: Optional[Path] = None,
    overwrite: bool = False,
):
    tic = time.monotonic()
    logging.info("Generating raw figures for subject: %s", sub)

    images = (
        index
        .filter("sub", sub)
        .filter("suffix", items={"T1w", "bold"})
        .filter("ext", items={".nii", ".nii.gz"})
    )
  # may have to modify filter here to get WM,GM,CSF images
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
            _participant_raw_t1w(
                record, out_dir, mriqc_dir=mriqc_dir, overwrite=overwrite
            )
        elif record["ent"]["suffix"] == "WM":
            _participant_brainmask_masked(
                record, out_dir, mriqc_dir=mriqc_dir, overwrite=overwrite
            )
        elif record["ent"]["suffix"] == "GM":
            _participant_brainmask_masked(
                record, out_dir, mriqc_dir=mriqc_dir, overwrite=overwrite
            )
        elif record["ent"]["suffix"] == "CSF":
            _participant_brainmask_masked(
                record, out_dir, mriqc_dir=mriqc_dir, overwrite=overwrite
            )
    logging.info(
        "Done processing subject: %s; elapsed: %.2fs", sub, time.monotonic() - tic
    )

def _participant_raw_t1w(
    record: pd.Series,
    out_dir: Path,
    mriqc_dir: Optional[Path] = None,
    overwrite: bool = False,
):
    entities = BIDSEntities.from_dict(record["ent"])

    img_path = Path(record["finfo"]["file_path"])
    logging.info("Processing: %s", img_path)
    img = nib.load(img_path)
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

    if mriqc_dir is not None and mriqc_dir.exists():
        _mriqc_metrics_tsv(record, entities, out_dir, mriqc_dir, overwrite=overwrite)

def _participant_brainmask_masked(
    record: pd.Series,
    out_dir: Path,
    mriqc_dir: Optional[Path],
    overwrite: bool = False,
):
    entities = BIDSEntities.from_dict(record["ent"])

    img_path = Path(record["finfo"]["file_path"])
    logging.info("Processing: %s", img_path)
    img = nib.load(img_path)
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

    if mriqc_dir is not None and mriqc_dir.exists():
        _mriqc_metrics_tsv(record, entities, out_dir, mriqc_dir, overwrite=overwrite)



