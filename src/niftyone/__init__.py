"""Large-scale neuroimaging visualization using FiftyOne."""

from ._version import __version__, __version_tuple__

# Register existing generators
from .figures.dwi import (
    DwiPerShellGenerator,
    QSpaceShellsGenerator,
    SignalPerVolumeGenerator,
)
from .figures.func import CarpetPlotGenerator, MeanStdGenerator
from .figures.multi_view import (
    SliceVideoGenerator,
    ThreeViewGenerator,
    ThreeViewVideoGenerator,
)
from .runner import Runner
