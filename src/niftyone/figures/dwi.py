"""Factories associated with diffusion visualizations."""

from niclips.figures import dwi
from niftyone.figures.factory import View, register


@register("qspace_shells")
class QSpaceShells(View):
    """Visualize diffusion gradients in Q-space."""

    entities = {"ext": ".mp4", "figure": "qspace"}
    view_fn = staticmethod(dwi.visualize_qspace)


@register("three_view_shell_video")
class DwiPerShell(View):
    """Visualize diffusion in ascending order of diffusion gradient strength."""

    entities = {"ext": ".mp4", "figure": "bval"}
    view_fn = staticmethod(dwi.three_view_per_shell)


@register("signal_per_volume")
class SignalPerVolume(View):
    """Visualize average signal per volume."""

    entities = {"ext": ".mp4", "figure": "signalPerVolume"}
    view_fn = staticmethod(dwi.signal_per_volume)
