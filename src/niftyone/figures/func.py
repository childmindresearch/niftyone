"""Factories associated with functional visualizations."""

from niclips.figures import bold
from niftyone.figures.factory import View, register


@register("carpet_plot")
class CarpetPlot(View):
    """Visualize carpet plots of functional data."""

    entities = {"ext": ".png", "figure": "carpet"}
    view_fn = staticmethod(bold.carpet_plot)


@register("mean_std")
class MeanStd(View):
    """Visualize mean and standard deviations of functional data."""

    entities = {"ext": ".png", "figure": "meanStd"}
    view_fn = staticmethod(bold.bold_mean_std)
