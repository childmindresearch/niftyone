"""Large-scale neuroimaging visualization using FiftyOne."""

from niftyone._version import __version__, __version_tuple__

# Register existing views
from niftyone.figures.dwi import (
    DwiPerShell,
    QSpaceShells,
    SignalPerVolume,
)
from niftyone.figures.func import CarpetPlot, MeanStd
from niftyone.figures.multi_view import (
    SliceVideo,
    ThreeView,
    ThreeViewVideo,
)
from niftyone.runner import Runner
