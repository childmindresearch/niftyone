"""Generators associated with diffusion data."""

from pathlib import Path

import pandas as pd

from niclips.figures import dwi
from niftyone.figures.generator import ViewGenerator, register


@register("qspace_shells")
class QSpaceShellsGenerator(ViewGenerator):
    def generate(self, record: pd.Series, out_dir: Path, overwrite: bool) -> None:
        self.generate_common(
            record=record,
            out_dir=out_dir,
            overwrite=overwrite,
            desc="qspace",
            ext=".mp4",
            view_fn=dwi.visualize_qspace,
        )
