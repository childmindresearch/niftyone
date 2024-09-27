"""Factories associated with functional data."""

from niclips.figures import bold
from niftyone.figures.factory import View


class CarpetPlot(View):
    entities = {"ext": ".png", "figure": "carpet"}
    view_fn = staticmethod(bold.carpet_plot)


class MeanStd(View):
    entities = {"ext": ".png", "figure": "meanStd"}
    view_fn = staticmethod(bold.bold_mean_std)
