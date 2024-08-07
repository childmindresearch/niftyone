"""Generator classes for different views."""

import ast
import inspect
import logging
import re
from abc import ABC
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

import nibabel as nib
import pandas as pd
from bids2table import BIDSEntities, BIDSTable
from matplotlib.figure import Figure
from PIL.Image import Image

import niclips.image as noimg
from niclips.typing import get_union_subclass

T = TypeVar("T", bound="ViewGenerator")

generator_registry: dict[str, type["ViewGenerator"]] = {}


def register(name: str) -> Callable[[type[T]], type[T]]:
    """Function to register generator to registry."""

    def decorator(cls: type[T]) -> type[T]:
        generator_registry[name] = cls
        return cls

    return decorator


def create_generator(view: str, query: str) -> "ViewGenerator":
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
        generator_instance = generator_cls(query, view_kwargs)
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


class ViewGenerator(ABC, Generic[T]):
    """Base view generator class."""

    entities: dict[str, Any] | None = None
    view_fn: Callable[[nib.Nifti1Image, Path], Image | Figure | None] | None = None

    def __init__(self, query: str, view_kwargs: dict[str, Any]) -> None:
        self.query: str = query
        self.view_kwargs: dict[str, Any] = view_kwargs

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

    def _figure_name(self) -> None:
        """Helper function to grab figure entity in view kwarg and update entities."""
        if "figure" in self.view_kwargs:
            assert self.entities
            self.entities["extra_entities"]["figure"] = self.view_kwargs["figure"]
            del self.view_kwargs["figure"]

    def generate(self, record: pd.Series, out_dir: Path, overwrite: bool) -> None:
        """Main call for generating view."""
        if not self.view_fn:
            raise ValueError("View is not provided, unable to create generator.")

        # Update figure name if necessary
        self._figure_name()

        img_path = Path(record["finfo"]["file_path"])
        logging.info("Processing: %s", img_path)

        signature = inspect.signature(self.view_fn)
        view_fn_input_type = signature.parameters[
            list(signature.parameters.keys())[0]
        ].annotation

        if get_union_subclass(view_fn_input_type, Path):
            img = img_path
        else:
            img = nib.nifti1.load(img_path)
            img = noimg.to_iso_ras(img)

        existing_entities = BIDSEntities.from_dict(record["ent"])
        out_path = existing_entities.with_update(self.entities).to_path(prefix=out_dir)
        out_path.parent.mkdir(exist_ok=True, parents=True)

        if not out_path.exists() or overwrite:
            logging.info("Generating %s", out_path)
            self.view_fn(img, out_path, **self.view_kwargs)
