"""Runner for running generator."""

import logging
from pathlib import Path

from bids2table import BIDSTable

from niftyone.figures.generator import ViewGenerator


class Runner:
    """Runner class for dynamic processing of participants."""

    def __init__(self, figure_generators: list[ViewGenerator]) -> None:
        self.figure_generators = figure_generators

    def gen_figures(
        self, table: BIDSTable, out_dir: Path, overwrite: bool = False
    ) -> None:
        if len(table) == 0:
            logging.info("Found no images")
            return

        logging.info(
            "Found %d images:\n\t%s",
            len(table),
            "\n\t".join(table.finfo["file_path"].tolist()),
        )
        for figure_generator in self.figure_generators:
            figure_generator(table=table, out_dir=out_dir, overwrite=overwrite)
