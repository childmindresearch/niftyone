"""Generators associated with functional data."""

from niclips.figures import bold
from niftyone.figures.factory import View, register


@register("carpet_plot")
class CarpetPlotGenerator(View):
    entities = {"ext": ".png", "extra_entities": {"figure": "carpet"}}
    view_fn = staticmethod(bold.carpet_plot)


@register("mean_std")
class MeanStdGenerator(View):
    entities = {"ext": ".png", "extra_entities": {"figure": "meanStd"}}
    view_fn = staticmethod(bold.bold_mean_std)
