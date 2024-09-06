"""Factory module for creating different figures."""

import logging
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

T = TypeVar("T", bound="View")

view_registry: dict[str, type["View"]] = {}


def register(name: str) -> Callable[[type[T]], type[T]]:
    """Function to register view to registry."""

    def decorator(cls: type[T]) -> type[T]:
        view_registry[name] = cls
        return cls

    return decorator


def create_view(
    view: str,
    view_kwargs: dict[str, Any] | None,
    join_entities: list[str],
    queries: list[str],
) -> "View":
    """Function to create view."""
    if view_kwargs is None:
        view_kwargs = {}

    try:
        view_cls = view_registry[view]
        view_instance = view_cls(queries, join_entities, view_kwargs)
        return view_instance
    except KeyError:
        msg = f"Factory for '{view}' for not found in registry."
        raise KeyError(msg)


def create_views(config: dict[str, Any]) -> list["View"]:
    """Create selected views dynamically from config with default settings."""
    views: list["View"] = []

    for group in config.get("figures", {}).values():
        queries = group.get("queries", [])
        join_entities = group.get("join_entities", ["sub", "ses"])
        views_map = group.get("views", {})

        for view, view_kwargs in views_map.items():
            views.append(
                create_view(
                    view=view,
                    view_kwargs=view_kwargs,
                    join_entities=join_entities,
                    queries=queries,
                )
            )

    return views


class View(ABC, Generic[T]):
    """Base view class."""

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
                df.reset_index(names=f"index_{ii}", drop=False).loc[
                    :, [f"index_{ii}"] + self.join_entities
                ]
            )
            for ii, df in enumerate(query_dfs)
        ]
        joined = reduce(
            lambda left, right: pd.merge(
                left, right, on=self.join_entities, how="inner"
            ),
            indexed_dfs,
        )
        indices = [
            joined[col].values for col in joined.columns if col.startswith("index_")
        ]

        for inds in zip(*indices):
            records = [table.nested.loc[ind] for ind in inds]
            self.create(records=records, out_dir=out_dir, overwrite=overwrite)

    def _figure_name(self) -> None:
        """Helper function to grab figure entity in view kwarg and update entities."""
        if "figure" in self.view_kwargs:
            assert self.entities
            self.entities["extra_entities"]["figure"] = self.view_kwargs["figure"]
            del self.view_kwargs["figure"]

    def create(
        self,
        records: list[pd.Series],
        out_dir: Path,
        overwrite: bool,
    ) -> None:
        """Main call for creating view."""
        if not self.view_fn:
            raise ValueError("No view factory provided, unable to create view.")

        # Update figure name if necessary
        self._figure_name()

        img_path = Path(records[0]["finfo"]["file_path"])
        logging.info("Processing: %s", img_path)

        img = nib.nifti1.load(img_path)
        img = noimg.to_iso_ras(img)

        # Handle overlays - currently only handles 1, update to handle multiple
        if len(records) > 1:
            overlays: list[nib.Nifti1Image] = []
            for overlay_record in records[1:]:
                overlay_path = Path(overlay_record["finfo"]["file_path"])
                overlay = nib.nifti1.load(overlay_path)
                overlays.append(noimg.to_iso_ras(overlay))
                self.view_kwargs["overlay"] = overlays.pop()

        existing_entities = BIDSEntities.from_dict(records[0]["ent"])
        out_path = existing_entities.with_update(self.entities).to_path(prefix=out_dir)
        out_path.parent.mkdir(exist_ok=True, parents=True)

        if not out_path.exists() or overwrite:
            logging.info("Creating %s", out_path)
            self.view_fn(img, out_path, **self.view_kwargs)

        plt.close("all")
