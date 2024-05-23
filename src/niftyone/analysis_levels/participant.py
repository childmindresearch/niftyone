"""Participant-level."""

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pkg_resources  # type:ignore [import-untyped]
import yaml  # type:ignore [import-untyped]
from bids2table import BIDSTable, bids2table
from elbow.utils import cpu_count, setup_logging

from niftyone import Runner
from niftyone.figures import generator


def load_config(config: Path | None) -> dict[str, Any]:
    """Helper to load configuration file."""
    if not config:
        config = Path(
            pkg_resources.resource_filename("niftyone", "resources/config.yaml")
        )

    with open(config, "r") as fpath:
        contents = yaml.safe_load(fpath)

    return contents


def participant(
    bids_dir: Path,
    out_dir: Path,
    sub: str | None = None,
    index_path: Path | None = None,
    qc_dir: Path | None = None,
    config: Path | None = None,
    workers: int = 1,
    overwrite: bool = False,
    verbose: bool = False,
) -> None:
    """NiftyOne participant analysis level."""
    bids_dir = Path(bids_dir)
    out_dir = Path(out_dir)

    if qc_dir:
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
        f"\n\tconfig: {config}"
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

    logging.info("Creating figure generators")
    config: dict[str, Any] = load_config(config=config)
    figure_generators = generator.create_generators(config=config)
    runner = Runner(figure_generators=figure_generators)

    _worker = partial(
        _participant_worker,
        workers=workers,
        subs=subs,
        index=index,
        out_dir=out_dir,
        qc_dir=qc_dir,
        runner=runner,
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
    runner: Runner,
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
            runner=runner,
            overwrite=overwrite,
        )


def _participant_single(
    sub: str,
    index: BIDSTable,
    out_dir: Path,
    runner: Runner,
    qc_dir: Path | None = None,
    overwrite: bool = False,
) -> None:
    tic = time.monotonic()

    logging.info(f"Processing subject {sub}")

    images = index.filter("sub", sub).filter("ext", items={".nii", ".nii.gz"})

    runner.gen_figures(table=images, out_dir=out_dir, overwrite=overwrite)

    logging.info(
        "Done processing subject: %s; elapsed: %.2fs", sub, time.monotonic() - tic
    )
