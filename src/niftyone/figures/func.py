"""Generators associated with multi-view."""

from pathlib import Path

import pandas as pd

from niclips.figures import bold
from niftyone.figures.generator import ViewGenerator, register


@register("carpet_plot")
class CarpetPlotGenerator(ViewGenerator):
    def generate(self, record: pd.Series, out_dir: Path, overwrite: bool) -> None:
        self.generate_common(
            record=record,
            out_dir=out_dir,
            overwrite=overwrite,
            entities={"desc": "carpet", "ext": ".png"},
            view_fn=bold.carpet_plot,
        )


@register("mean_std")
class MeanStdGenerator(ViewGenerator):
    def generate(self, record: pd.Series, out_dir: Path, overwrite: bool) -> None:
        self.generate_common(
            record=record,
            out_dir=out_dir,
            overwrite=overwrite,
            entities={"desc": "meanStd", "ext": ".png"},
            view_fn=bold.bold_mean_std,
        )
