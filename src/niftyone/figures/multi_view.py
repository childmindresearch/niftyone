"""Generators associated with multi-view."""

from niclips.figures import multi_view
from niftyone.figures.generator import ViewGenerator, register


@register("three_view")
class ThreeViewGenerator(ViewGenerator):
    entities = {"ext": ".png", "extra_entities": {"figure": "threeView"}}
    view_fn = staticmethod(multi_view.three_view_frame)


@register("slice_video")
class SliceVideoGenerator(ViewGenerator):
    entities = {"ext": ".mp4", "extra_entities": {"figure": "sliceVideo"}}
    view_fn = staticmethod(multi_view.slice_video)


@register("three_view_overlay")
class ThreeViewOverlayGenerator(ViewGenerator):
    entities = {"desc": "threeViewOverlay", "ext": ".png"}
    view_fn = staticmethod(multi_view.three_view_overlay_frame)


@register("three_view_video")
class ThreeViewVideoGenerator(ViewGenerator):
    entities = {"ext": ".mp4", "extra_entities": {"figure": "threeViewVideo"}}
    view_fn = staticmethod(multi_view.three_view_video)
