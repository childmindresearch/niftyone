"""Generators associated with diffusion data."""

from niclips.figures import dwi
from niftyone.figures.generator import ViewGenerator, register


@register("qspace_shells")
class QSpaceShellsGenerator(ViewGenerator):
    entities = {"desc": "qspace", "ext": ".mp4"}
    view_fn = staticmethod(dwi.visualize_qspace)


@register("three_view_shell_video")
class DwiPerShellGenerator(ViewGenerator):
    entities = {"desc": "bval", "ext": ".mp4"}
    view_fn = staticmethod(dwi.three_view_per_shell)
