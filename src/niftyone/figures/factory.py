"""Factory module for creating different figures."""

import logging
from abc import ABC
from functools import reduce
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Generic, TypeVar

import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
from bids2table import BIDSEntities, BIDSTable

import niclips.image as noimg

T = TypeVar("T", bound="View")

view_registry: dict[str, type["View"]] = {}


def register(name: str) -> Callable:
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
    view_kwargs = view_kwargs or {}
    try:
        view_cls = view_registry[view]
        return view_cls(queries, join_entities, view_kwargs)
    except KeyError:
        raise KeyError(f"Factory for '{view}' for not found in registry.")


def create_views(config: dict[str, Any]) -> list["View"]:
    """Create selected views dynamically from config with default settings."""
    return [
        create_view(
            view=view,
            view_kwargs=view_kwargs,
            join_entities=group.get("join_entities", ["sub", "ses"]),
            queries=group.get("queries", []),
        )
        for group in config.get("figures", {}).values()
        for view, view_kwargs in group.get("views", {}).items()
    ]


class View(ABC, Generic[T]):
    """Base view class."""

    entities: dict[str, Any] | None = None
    view_fn: Callable | None = None

    def __init__(
        self,
        queries: list[str],
        join_entities: list[str] | None,
        view_kwargs: dict[str, Any],
    ) -> None:
        self.queries = queries
        self.view_kwargs = MappingProxyType(view_kwargs)  # Immutable dict
        self.join_entities = join_entities or []

    def __call__(
        self,
        table: BIDSTable,
        out_dir: Path,
        overwrite: bool,
    ) -> None:
        # Filters by entities via string query
        # First query is for main image, subsequent are for overlays
        query_dfs = [table.ent.query(q) for q in self.queries]
        indexed_dfs = [
            df.reset_index(names=f"index_{ii}", drop=False)[
                [f"index_{ii}"] + self.join_entities
            ]
            for ii, df in enumerate(query_dfs)
        ]
        joined = reduce(
            lambda left, right: pd.merge(
                left, right, on=self.join_entities, how="inner"
            ),
            indexed_dfs,
        )
        indices = (joined[col].values for col in joined if col.startswith("index_"))

        for inds in zip(*indices):
            records = [table.nested.loc[ind] for ind in inds]
            self.create(records=records, out_dir=out_dir, overwrite=overwrite)

    def _load_image(self, record: pd.Series, log: bool = False) -> nib.Nifti1Image:
        """Helper to load image."""
        img_path = Path(record["finfo"]["file_path"])
        if log:
            logging.info("Processing %s", img_path)
        img = nib.nifti1.load(img_path)

        return noimg.to_iso_ras(img)

    def _load_overlays(self, overlay_records: list[pd.Series]) -> list[nib.Nifti1Image]:
        """Helper to load overlays."""
        return [
            self._load_image(record=overlay_record)
            for overlay_record in overlay_records
        ]

    def _figure_out_path(self, record: pd.Series, out_dir: Path) -> Path:
        """Generates the output figure file path."""
        figure_value = self.view_kwargs.get("figure")
        figure_entities = {
            **record["ent"].to_dict(),
            **(self.entities if self.entities is not None else {}),
            **({"figure": figure_value} if figure_value is not None else {}),
        }

        out_path = BIDSEntities.from_dict(figure_entities).to_path(prefix=out_dir)
        out_path.parent.mkdir(exist_ok=True, parents=True)
        return out_path

    def create(
        self,
        records: list[pd.Series],
        out_dir: Path,
        overwrite: bool,
    ) -> None:
        """Main call for creating view."""
        if not self.view_fn:
            raise ValueError("No view factory provided, unable to create view.")

        img = self._load_image(record=records[0], log=True)
        overlays = (
            self._load_overlays(overlay_records=records[1:])
            if len(records) > 1
            else None
        )
        out_path = self._figure_out_path(records[0], out_dir)

        if not out_path.exists() or overwrite:
            logging.info("Creating %s", out_path)
            self.view_fn(img, out_path, overlays=overlays, **self.view_kwargs)

        plt.close("all")
