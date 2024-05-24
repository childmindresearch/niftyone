"""Generator classes for different views."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import nibabel as nib
import pandas as pd
from bids2table import BIDSEntities, BIDSTable
from matplotlib.figure import Figure
from PIL.Image import Image

import niclips.image as noimg


@dataclass()
class ViewGenerator(ABC):
    """Base view generator class."""

    query: str
    view_kwargs: dict[str, Any]

    def __call__(
        self,
        table: BIDSTable,
        out_dir: Path,
        overwrite: bool,
    ) -> None:
        # Filters by entity (via string query)
        table_idxes = table.ent.query(self.query).index
        for idx in table_idxes:
            record = table.nested.loc[idx]
            self.generate(record=record, out_dir=out_dir, overwrite=overwrite)

    def generate_common(
        self,
        record: pd.Series,
        out_dir: Path,
        overwrite: bool,
        desc: str,
        ext: str,
        view_fn: Callable[[nib.nifti1.Nifti1Image, Path], Image | Figure | None],
    ) -> None:
        """Partial function for calling generate method."""
        img_path = Path(record["finfo"]["file_path"])
        logging.info("Processing: %s", img_path)
        img = nib.nifti1.load(img_path)
        img = noimg.to_iso_ras(img)

        entities = BIDSEntities.from_dict(record["ent"])
        out_path = entities.with_update(desc=desc, ext=ext).to_path(prefix=out_dir)
        out_path.parent.mkdir(exist_ok=True, parents=True)
        if not out_path.exists() or overwrite:
            logging.info("Generating %s", out_path)
            view_fn(img, out_path, **self.view_kwargs)

    @abstractmethod
    def generate(self, record: pd.Series, out_dir: Path, overwrite: bool) -> None:
        """Main call for generating view."""
        pass
