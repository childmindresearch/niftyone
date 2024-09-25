"""Factories associated with diffusion data."""

from niclips.figures import dwi
from niftyone.figures.factory import View, register


@register("qspace_shells")
class QSpaceShells(View):
    entities = {"ext": ".mp4", "figure": "qspace"}
    view_fn = staticmethod(dwi.visualize_qspace)


@register("three_view_shell_video")
class DwiPerShell(View):
    entities = {"ext": ".mp4", "figure": "bval"}
    view_fn = staticmethod(dwi.three_view_per_shell)


@register("signal_per_volume")
class SignalPerVolume(View):
    entities = {"ext": ".mp4", "figure": "signalPerVolume"}
    view_fn = staticmethod(dwi.signal_per_volume)
