"""Generators associated with multi-view."""

from pathlib import Path

import pandas as pd

from niclips.figures import multi_view
from niftyone.figures.generator import ViewGenerator


class ThreeViewGenerator(ViewGenerator):
    def generate(self, record: pd.Series, out_dir: Path, overwrite: bool) -> None:
        self.generate_common(
            record=record,
            out_dir=out_dir,
            overwrite=overwrite,
            desc="threeView",
            ext=".png",
            view_fn=multi_view.three_view_frame,
        )


class SliceVideoGenerator(ViewGenerator):
    def generate(self, record: pd.Series, out_dir: Path, overwrite: bool) -> None:
        self.generate_common(
            record=record,
            out_dir=out_dir,
            overwrite=overwrite,
            desc="sliceVideo",
            ext=".mp4",
            view_fn=multi_view.slice_video,
        )


class ThreeViewVideoGenerator(ViewGenerator):
    """Three view video generator (4D over time)."""

    def generate(self, record: pd.Series, out_dir: Path, overwrite: bool) -> None:
        self.generate_common(
            record=record,
            out_dir=out_dir,
            overwrite=overwrite,
            desc="threeViewVideo",
            ext=".mp4",
            view_fn=multi_view.three_view_video,
        )
