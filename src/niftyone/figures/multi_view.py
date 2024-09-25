"""Factories associated with multi-view."""

from niclips.figures import multi_view
from niftyone.figures.factory import View, register


@register("three_view")
class ThreeView(View):
    entities = {"ext": ".png", "extra_entities": {"figure": "threeView"}}
    view_fn = staticmethod(multi_view.three_view_frame)


@register("slice_video")
class SliceVideo(View):
    entities = {"ext": ".mp4", "extra_entities": {"figure": "sliceVideo"}}
    view_fn = staticmethod(multi_view.slice_video)


@register("three_view_video")
class ThreeViewVideo(View):
    entities = {"ext": ".mp4", "extra_entities": {"figure": "threeViewVideo"}}
    view_fn = staticmethod(multi_view.three_view_video)
