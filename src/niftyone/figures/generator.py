"""Generator classes for different views."""

import ast
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

import nibabel as nib
import pandas as pd
from bids2table import BIDSEntities, BIDSTable
from matplotlib.figure import Figure
from PIL.Image import Image

import niclips.image as noimg

T = TypeVar("T", bound="ViewGenerator")

generator_registry: dict[str, type["ViewGenerator"]] = {}


def register(name: str) -> Callable[[type[T]], type[T]]:
    """Function to register generator to registry."""

    def decorator(cls: type[T]) -> type[T]:
        generator_registry[name] = cls
        return cls

    return decorator


def create_generator(view: str, query: str, **kwargs) -> "ViewGenerator":
    """Function to create generator."""

    def _parse_view(view: str) -> tuple[str, dict[str, Any]]:
        """Parse view for figure-specific kwargs."""
        # Match [view][(key1=value1,key2=value2,...)]
        match = re.match(r"(\w+)\(([^=,]+=[^=,]+(?:,[^=,]+=[^=,]+)*)\)", view)

        if match:
            view = match.group(1)
            view_kwargs = match.group(2)

            expr = ast.parse(f"fn({view_kwargs})", mode="eval")
            view_kwargs = {
                keyword.arg: ast.literal_eval(keyword.value)
                for keyword in expr.body.keywords  # type: ignore [attr-defined]
            }
            return view, view_kwargs
        else:
            return view, {}

    view, view_kwargs = _parse_view(view)
    try:
        generator_cls = generator_registry[view]
        generator_instance = generator_cls(query, view_kwargs, **kwargs)
        return generator_instance
    except KeyError:
        msg = f"Generator for '{view}' for not found in registry."
        raise KeyError(msg)


def create_generators(config: dict[str, Any]) -> list["ViewGenerator"]:
    """Create selected generators dynamically from config with default settings."""
    generators: list["ViewGenerator"] = []

    for group in config.values():
        query = group.get("query", "")
        views = group.get("views", [])

        for view in views:
            generators.append(create_generator(view=view, query=query))

    return generators


@dataclass()
class ViewGenerator(ABC, Generic[T]):
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
        entities: dict[str, str],
        view_fn: Callable[[nib.nifti1.Nifti1Image, Path], Image | Figure | None],
    ) -> None:
        """Partial function for calling generate method."""
        img_path = Path(record["finfo"]["file_path"])
        logging.info("Processing: %s", img_path)
        img = nib.nifti1.load(img_path)
        img = noimg.to_iso_ras(img)

        existing_entities = BIDSEntities.from_dict(record["ent"])
        out_path = existing_entities.with_update(entities).to_path(prefix=out_dir)
        out_path.parent.mkdir(exist_ok=True, parents=True)
        if not out_path.exists() or overwrite:
            logging.info("Generating %s", out_path)
            view_fn(img, out_path, **self.view_kwargs)

    @abstractmethod
    def generate(self, record: pd.Series, out_dir: Path, overwrite: bool) -> None:
        """Main call for generating view."""
        pass
