"""Factories associated with diffusion data."""

from niclips.figures import dwi
from niftyone.figures.factory import View


class QSpaceShells(View):
    entities = {"ext": ".mp4", "figure": "qspace"}
    view_fn = staticmethod(dwi.visualize_qspace)
    view_name = "qspace_shells"


class DwiPerShell(View):
    entities = {"ext": ".mp4", "figure": "bval"}
    view_fn = staticmethod(dwi.three_view_per_shell)
    view_name = "three_view_shell_video"


class SignalPerVolume(View):
    entities = {"ext": ".mp4", "figure": "signalPerVolume"}
    view_fn = staticmethod(dwi.signal_per_volume)
    view_name = "signal_per_volume"
