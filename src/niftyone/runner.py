"""Runner for creating figures and extracting QC metrics."""

import logging
from pathlib import Path

from bids2table import BIDSEntities, BIDSTable

from niftyone.figures.factory import View
from niftyone.metrics import create_niftyone_metrics_tsv


class Runner:
    """Runner class for dynamic processing of participants."""

    table: BIDSTable

    def __init__(
        self,
        figure_views: list[View],
        out_dir: Path,
        qc_dir: Path | None,
        overwrite: bool,
    ) -> None:
        self.figure_views = figure_views
        self.out_dir = out_dir
        self.qc_dir = qc_dir
        self.overwrite = overwrite

    def create_figures(self) -> None:
        """Function to create figures."""
        images = self.table.filter("ext", items={".nii", ".nii.gz"})
        if (num_images := len(images)) == 0:
            logging.info("Found no images")
            return

        logging.info(
            "Found %d images:\n\t%s",
            num_images,
            "\n\t".join(self.table.finfo["file_path"].tolist()),
        )
        for figure_view in self.figure_views:
            figure_view(table=images, out_dir=self.out_dir, overwrite=self.overwrite)

    def update_metrics(self) -> None:
        """Function to create / update QC metrics.

        NOTE: Writing (and later reading) individual metric files for each
        unique combination of entities. Look into possibility of single file
        for each modality or suffix.
        """
        # If no qc_dir provided
        if not self.qc_dir:
            return

        images = self.table.filter("ext", items={".nii.gz", ".nii"})
        for _, record in images.nested.iterrows():
            create_niftyone_metrics_tsv(
                record=record,
                entities=BIDSEntities.from_dict(record["ent"]),
                out_dir=self.out_dir,
                qc_dir=self.qc_dir,
                overwrite=self.overwrite,
            )
