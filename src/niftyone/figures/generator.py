"""Generator classes for different views."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, Type, TypeVar

import nibabel as nib
import pandas as pd
from bids2table import BIDSEntities, BIDSTable
from matplotlib.figure import Figure
from PIL.Image import Image

import niclips.image as noimg

generator_registry: dict[str, Type["ViewGenerator"]] = {}

T = TypeVar("T", bound="ViewGenerator")


def register(name: str) -> Callable[[Type[T]], Type[T]]:
    """Function to register generator to registry."""

    def decorator(cls: Type[T]) -> Type[T]:
        generator_registry[name] = cls
        return cls

    return decorator


def create_generators(
    config: dict[str, Any],
) -> list["ViewGenerator"]:
    """Function to create generators dynamically from config."""
    generators = []

    for settings in config.values():
        query = settings.get("query", "")
        views = settings.get("views", [])

        for view in views:
            if view in generator_registry:
                generator_cls = generator_registry[view]
                generator_instance = generator_cls(query=query)
                generators.append(generator_instance)
            else:
                logging.warning(f"Generator for '{view}' not found in registry")

    return generators


@dataclass
class ViewGenerator(ABC, Generic[T]):
    """Base view generator class."""

    query: str

    def __call__(self, table: BIDSTable, out_dir: Path, overwrite: bool) -> None:
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
            view_fn(img, out_path)

    @abstractmethod
    def generate(self, record: pd.Series, out_dir: Path, overwrite: bool) -> None:
        """Main call for generating view."""
        pass
