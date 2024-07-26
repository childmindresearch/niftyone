"""Generators associated with multi-view."""

from niclips.figures import multi_view
from niftyone.figures.generator import ViewGenerator, register


@register("three_view")
class ThreeViewGenerator(ViewGenerator):
    entities = {"desc": "threeView", "ext": ".png"}
    view_fn = staticmethod(multi_view.three_view_frame)


@register("slice_video")
class SliceVideoGenerator(ViewGenerator):
    entities = {"desc": "sliceVideo", "ext": ".mp4"}
    view_fn = staticmethod(multi_view.slice_video)


@register("three_view_video")
class ThreeViewVideoGenerator(ViewGenerator):
    entities = {"desc": "threeViewVideo", "ext": ".mp4"}
    view_fn = staticmethod(multi_view.three_view_video)
