"""Factories associated with functional data."""

from niclips.figures import bold
from niftyone.figures.factory import View, register


@register("carpet_plot")
class CarpetPlot(View):
    entities = {"ext": ".png", "figure": "carpet"}
    view_fn = staticmethod(bold.carpet_plot)


@register("mean_std")
class MeanStd(View):
    entities = {"ext": ".png", "figure": "meanStd"}
    view_fn = staticmethod(bold.bold_mean_std)
