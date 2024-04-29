import numpy as np
import pytest
from PIL import Image, ImageFont

from niclips.image._annotate import _get_font, annotate


class TestAnnotate:
    def test_annotate_array(self, img_array: np.ndarray):
        new_img = annotate(img=img_array, text="Test", loc="upper left")

        assert isinstance(new_img, Image.Image)

    def test_inplace_true(self, img_pil: Image.Image):
        new_img = annotate(img=img_pil, text="Test", loc="upper left", inplace=True)

        assert new_img is img_pil

    def test_inplace_false(self, img_pil: Image.Image):
        new_img = annotate(img=img_pil, text="Test", loc="upper left", inplace=False)

        assert new_img is not img_pil

    @pytest.mark.parametrize(
        ("loc"),
        [("upper left"), ("upper right"), ("lower left"), ("lower right")],
    )
    def test_valid_locs(self, img_pil: Image.Image, loc: str):
        img = annotate(img=img_pil, text="Text", loc=loc)
        assert isinstance(img, Image.Image)

    def test_invalid_loc(self, img_pil: Image.Image):
        with pytest.raises(ValueError, match="Unsupported loc.*"):
            annotate(img=img_pil, text="Text", loc="Test")


def test_get_font():
    font = _get_font(size=12)
    assert isinstance(font, ImageFont.FreeTypeFont)
