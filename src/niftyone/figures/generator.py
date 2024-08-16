"""Generator classes for different views."""

import ast
import logging
import re
from abc import ABC
from functools import reduce
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

import matplotlib.pyplot as plt
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


def create_generator(
    view: str, join_entities: list[str], queries: list[str]
) -> "ViewGenerator":
    """Function to create generator."""

    def _parse_view(view: str) -> tuple[str, dict[str, Any]]:
        """Parse view for figure-specific kwargs."""
        # Match [view][(key1=value1,key2=value2,key3=entities_dict,...)]
        match = re.match(r"(\w+)\(([^()]+)\)", view)

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
        generator_instance = generator_cls(queries, join_entities, view_kwargs)
        return generator_instance
    except KeyError:
        msg = f"Generator for '{view}' for not found in registry."
        raise KeyError(msg)


def create_generators(config: dict[str, Any]) -> list["ViewGenerator"]:
    """Create selected generators dynamically from config with default settings."""
    generators: list["ViewGenerator"] = []

    for group in config.get("figures", None).values():
        queries = group.get("queries", "")
        join_entities = group.get("join_entities", ["sub", "ses"])
        views = group.get("views", [])

        for view in views:
            generators.append(
                create_generator(
                    view=view, join_entities=join_entities, queries=queries
                )
            )

    return generators


class ViewGenerator(ABC, Generic[T]):
    """Base view generator class."""

    entities: dict[str, Any] | None = None
    view_fn: Callable[[nib.Nifti1Image, Path], Image | Figure | None] | None = None

    def __init__(
        self,
        queries: list[str],
        join_entities: list[str] | None,
        view_kwargs: dict[str, Any],
    ) -> None:
        self.queries = queries
        self.view_kwargs = view_kwargs
        self.join_entities = join_entities or []

    def __call__(
        self,
        table: BIDSTable,
        out_dir: Path,
        overwrite: bool,
    ) -> None:
        # Filters by entity (via string query)
        # First query is for main image, subsequent are for overlays
        query_dfs = [table.ent.query(q) for q in self.queries]
        indexed_dfs = [
            (
                df.assign(table_index=df.index)
                .reset_index(drop=True)
                .loc[:, ["table_index"] + self.join_entities]
                .rename(columns={"table_index": f"index_{ii}"})
            )
            for ii, df in enumerate(query_dfs)
        ]
        joined = reduce(
            lambda left, right: pd.merge(
                left, right, on=self.join_entities, how="outer"
            ),
            indexed_dfs,
        )
        indices = [joined[f"index_{ii}"].values for ii in range(len(self.queries))]

        records = [table.nested.loc[ind] for inds in zip(*indices) for ind in inds]
        self.generate(records=records, out_dir=out_dir, overwrite=overwrite)

    def _figure_name(self) -> None:
        """Helper function to grab figure entity in view kwarg and update entities."""
        if "figure" in self.view_kwargs:
            assert self.entities
            self.entities["extra_entities"]["figure"] = self.view_kwargs["figure"]
            del self.view_kwargs["figure"]

    def generate(
        self,
        records: list[pd.Series],
        out_dir: Path,
        overwrite: bool,
    ) -> None:
        """Main call for generating view."""
        if not self.view_fn:
            raise ValueError("View is not provided, unable to create generator.")

        # Update figure name if necessary
        self._figure_name()

        img_path = Path(records[0]["finfo"]["file_path"])
        logging.info("Processing: %s", img_path)

        img = nib.nifti1.load(img_path)
        img = noimg.to_iso_ras(img)

        # Handle overlays - currently only handles 1, update to handle multiple
        overlays = []
        # Temporary logic to handle multiple overlays
        if len(records) > 2:
            raise NotImplementedError("Multi-image overlay not yet implemented")

        if len(records) > 1:
            for overlay_record in records[1:]:
                overlay_path = Path(overlay_record["finfo"]["file_path"])
                overlay = nib.nifti1.load(overlay_path)
                overlays.append(noimg.to_iso_ras(overlay))
                self.view_kwargs["overlay"] = overlays.pop()

        existing_entities = BIDSEntities.from_dict(records[0]["ent"])
        out_path = existing_entities.with_update(self.entities).to_path(prefix=out_dir)
        out_path.parent.mkdir(exist_ok=True, parents=True)

        if not out_path.exists() or overwrite:
            logging.info("Generating %s", out_path)
            self.view_fn(img, out_path, **self.view_kwargs)

        plt.close("all")
