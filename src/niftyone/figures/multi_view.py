"""Factories associated with multiple views."""

from niclips.figures import multi_view
from niftyone.figures.factory import View, register


@register("three_view")
class ThreeView(View):
    """Visualize static image in three orientations (coronal, axial, sagittal)."""

    entities = {"ext": ".png", "figure": "threeView"}
    view_fn = staticmethod(multi_view.three_view_frame)


@register("slice_video")
class SliceVideo(View):
    """Visualize along single chosen orientation of an image volume."""

    entities = {"ext": ".mp4", "figure": "sliceVideo"}
    view_fn = staticmethod(multi_view.slice_video)


@register("three_view_video")
class ThreeViewVideo(View):
    """Visualize image volume along three orientations (coronal, axial, sagittal)."""

    entities = {"ext": ".mp4", "figure": "threeViewVideo"}
    view_fn = staticmethod(multi_view.three_view_video)
