"""Large-scale neuroimaging visualization using FiftyOne."""

from ._version import __version__, __version_tuple__

# Register existing generators
from .figures.func import CarpetPlotGenerator, MeanStdGenerator
from .figures.multi_view import (
    SliceVideoGenerator,
    ThreeViewGenerator,
    ThreeViewVideoGenerator,
)
