"""Runner for running generator."""

from pathlib import Path

from bids2table import BIDSTable

from niftyone.figures.generator import ViewGenerator


class Runner:
    """Runner class for generating figures."""

    def __init__(self, generators: list[ViewGenerator]) -> None:
        self.generators = generators

    def run(self, table: BIDSTable, out_dir: Path, overwrite: bool = False) -> None:
        for generator in self.generators:
            generator(table=table, out_dir=out_dir, overwrite=overwrite)
