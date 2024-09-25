"""Large-scale neuroimaging visualization using FiftyOne."""

# Import all plugins. In particular this should register user's custom generators
from . import plugins
from ._version import __version__, __version_tuple__

# Register existing views
from .figures.dwi import (
    DwiPerShell,
    QSpaceShells,
    SignalPerVolume,
)
from .figures.func import CarpetPlot, MeanStd
from .figures.multi_view import (
    SliceVideo,
    ThreeView,
    ThreeViewVideo,
)
from .runner import Runner
