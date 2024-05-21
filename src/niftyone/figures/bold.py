"""Generate pipeline figures for bold images."""

import logging
from pathlib import Path

import nibabel as nib
import pandas as pd
from bids2table.entities import BIDSEntities
from matplotlib import pyplot as plt

import niclips.image as noimg
from niclips.figures import bold, multi_view
from niftyone import metrics


def raw_bold(
    record: pd.Series,
    out_dir: Path,
    qc_dir: Path | None = None,
    overwrite: bool = False,
) -> None:
    """Generate figures for raw participant bold."""
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
        multi_view.three_view_video(img, out=out_path)

    out_path = entities.with_update(desc="carpet", ext=".png").to_path(prefix=out_dir)
    if not out_path.exists() or overwrite:
        logging.info("Generating: %s", out_path)
        fig = bold.carpet_plot(img, out=out_path)
        plt.close(fig)

    out_path = entities.with_update(desc="meanStd", ext=".png").to_path(prefix=out_dir)
    if not out_path.exists() or overwrite:
        logging.info("Generating: %s", out_path)
        bold.bold_mean_std(img, out=out_path)

    if qc_dir is not None and qc_dir.exists():
        metrics.gen_niftyone_metrics_tsv(
            record, entities, out_dir, qc_dir, overwrite=overwrite
        )
