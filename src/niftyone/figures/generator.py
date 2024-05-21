"""Generator classes for different views."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Type, TypeVar

import nibabel as nib
import pandas as pd
from bids2table.entities import BIDSEntities
from matplotlib.figure import Figure
from PIL.Image import Image

import niclips.image as noimg

GENERATOR_REGISTRY: dict[str, Type["ViewGenerator"]] = {}

T = TypeVar("T", bound="ViewGenerator")


def register(name: str) -> Callable[[Type[T]], Type[T]]:
    """Function to register generator to registry."""

    def decorator(cls: Type[T]) -> Type[T]:
        GENERATOR_REGISTRY[name] = cls
        return cls

    return decorator


@dataclass
class ViewGenerator(ABC, Generic[T]):
    """Base view generator class."""

    query: str

    def __call__(self, table: pd.DataFrame, out_dir: Path, overwrite: bool) -> None:
        filtered_table = table.query(self.query)
        for _, record in filtered_table.iterrows():
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
            view_fn(img, out_path)

    @abstractmethod
    def generate(self, record: pd.Series, out_dir: Path, overwrite: bool) -> None:
        """Main call for generating view."""
        pass
