"""Large-scale neuroimaging visualization using FiftyOne."""

from pathlib import Path

from ._version import __version__, __version_tuple__
from .figures.factory import __file__ as factory_path
from .figures.factory import register_views
from .runner import Runner

# Register all default views
register_views(search_path=str(Path(factory_path).parent))
