"""Generators associated with diffusion data."""

from niclips.figures import dwi
from niftyone.figures.generator import ViewGenerator, register


@register("qspace_shells")
class QSpaceShellsGenerator(ViewGenerator):
    entities = {"ext": ".mp4", "extra_entities": {"figure": "qspace"}}
    view_fn = staticmethod(dwi.visualize_qspace)


@register("three_view_shell_video")
class DwiPerShellGenerator(ViewGenerator):
    entities = {"ext": ".mp4", "extra_entities": {"figure": "bval"}}
    view_fn = staticmethod(dwi.three_view_per_shell)


@register("signal_per_volume")
class SignalPerVolumeGenerator(ViewGenerator):
    entities = {"ext": ".mp4", "extra_entities": {"figure": "signalPerVolume"}}
    view_fn = staticmethod(dwi.signal_per_volume)
