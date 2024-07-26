"""Generators associated with multi-view."""

from niclips.figures import bold
from niftyone.figures.generator import ViewGenerator, register


@register("carpet_plot")
class CarpetPlotGenerator(ViewGenerator):
    entities = {"desc": "carpet", "ext": ".png"}
    view_fn = staticmethod(bold.carpet_plot)


@register("mean_std")
class MeanStdGenerator(ViewGenerator):
    entities = {"desc": "meanStd", "ext": ".png"}
    view_fn = staticmethod(bold.bold_mean_std)
