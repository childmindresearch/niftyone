"""Global registry of figure generators."""

from .func import CarpetPlotGenerator, MeanStdGenerator
from .generator import ViewGenerator
from .multi_view import (
    SliceVideoGenerator,
    ThreeViewGenerator,
    ThreeViewVideoGenerator,
)

_ALL_VIEWS: dict[str, type[ViewGenerator]] = {
    "carpet_plot": CarpetPlotGenerator,
    "mean_std": MeanStdGenerator,
    "slice_video": SliceVideoGenerator,
    "three_view": ThreeViewGenerator,
    "three_view_video": ThreeViewVideoGenerator,
}
_REGISTRY: dict[str, type[ViewGenerator]] = {}


def register_generator(
    view: str, cls: type[ViewGenerator], overwrite: bool = False
) -> None:
    """Add a view generator class to the global registry."""
    if view in _REGISTRY and not overwrite:
        raise ValueError(
            f"Generator '{view}' already registered; use overwrite=True to overwrite."
        )
    _REGISTRY[view] = cls


def create_generator(view: str, query: str, **kwargs) -> "ViewGenerator":
    """Create view generator by looking up in global registry."""
    if view not in _REGISTRY:
        raise KeyError(f"Generator for '{view}' for not found in registry.")

    cls = _REGISTRY[view]
    return cls(query=query, **kwargs)


def list_generators() -> list[str]:
    """Return the list of registered views."""
    return list(_REGISTRY.keys())


def clear_registry() -> None:
    """Clear global view registry."""
    _REGISTRY.clear()


def register_all_views() -> None:
    """Register all available view generators."""
    clear_registry()
    for view, cls in _ALL_VIEWS.items():
        register_generator(view, cls)


register_all_views()
