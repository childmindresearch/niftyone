"""Runner for running generator."""

from pathlib import Path

from bids2table import BIDSTable

from niftyone.figures.generator import ViewGenerator


class Runner:
    """Runner class for generating figures."""

    def __init__(self, figure_generators: list[ViewGenerator]) -> None:
        self.figure_generators = figure_generators

    def run(self, table: BIDSTable, out_dir: Path, overwrite: bool = False) -> None:
        for figure_generator in self.figure_generators:
            figure_generator(table=table, out_dir=out_dir, overwrite=overwrite)
