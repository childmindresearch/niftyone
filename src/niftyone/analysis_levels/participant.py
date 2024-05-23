"""Participant-level."""

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import numpy as np
from bids2table import BIDSTable, bids2table
from elbow.utils import cpu_count, setup_logging

from niftyone import figures, typing


def participant(
    bids_dir: typing.StrPath,
    out_dir: typing.StrPath,
    sub: str | None = None,
    index_path: typing.StrPath | None = None,
    qc_dir: typing.StrPath | None = None,
    workers: int = 1,
    overwrite: bool = False,
    verbose: bool = False,
) -> None:
    """NiftyOne participant analysis level."""
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
        "Starting niftyone participant-level:"
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
        _participant_worker,
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


def _participant_worker(
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
        _participant_single(
            sub=sub,
            index=index,
            out_dir=out_dir,
            qc_dir=qc_dir,
            overwrite=overwrite,
        )


def _participant_single(
    sub: str,
    index: BIDSTable,
    out_dir: Path,
    qc_dir: Path | None = None,
    overwrite: bool = False,
) -> None:
    tic = time.monotonic()
    logging.info("Generating raw figures for subject: %s", sub)

    images = index.filter("sub", sub).filter("ext", items={".nii", ".nii.gz"})
    if len(images) == 0:
        logging.info("Found no images")
        return

    logging.info(
        "Found %d images:\n\t%s",
        len(images),
        "\n\t".join(images.finfo["file_path"].tolist()),
    )

    for _, record in images.nested.iterrows():
        match record["ent"]["suffix"]:
            case "T1w":
                figures.anat.raw_t1w(
                    record, out_dir, qc_dir=qc_dir, overwrite=overwrite
                )
            case "bold":
                figures.func.raw_bold(
                    record, out_dir, qc_dir=qc_dir, overwrite=overwrite
                )
            case _:
                logging.info(
                    f"Figure generation for {record['ent']['suffix']} not yet "
                    "implemented."
                )
    logging.info(
        "Done processing subject: %s; elapsed: %.2fs", sub, time.monotonic() - tic
    )
