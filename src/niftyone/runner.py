"""Runner for running generator."""

import logging
from pathlib import Path

from bids2table import BIDSTable

from niftyone.figures.generator import ViewGenerator


class Runner:
    """Runner class for dynamic processing of participants."""

    table: BIDSTable

    def __init__(
        self,
        figure_generators: list[ViewGenerator],
        out_dir: Path,
        qc_dir: Path | None,
        overwrite: bool,
    ) -> None:
        self.figure_generators = figure_generators
        self.out_dir = out_dir
        self.qc_dir = qc_dir
        self.overwrite = overwrite

    def gen_figures(self) -> None:
        images = self.table.filter("ext", items={".nii", ".nii.gz"})
        if (num_images := len(images)) == 0:
            logging.info("Found no images")
            return

        logging.info(
            "Found %d images:\n\t%s",
            num_images,
            "\n\t".join(self.table.finfo["file_path"].tolist()),
        )
        for figure_generator in self.figure_generators:
            figure_generator(
                table=images, out_dir=self.out_dir, overwrite=self.overwrite
            )
