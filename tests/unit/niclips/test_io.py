from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from av.video.stream import VideoStream
from PIL import Image

import niclips.io as noio


class TestVideoWriterClassInit:
    def test_init_non_mp4(self, tmp_path: Path):
        video_path = tmp_path / "video.avi"
        with pytest.raises(ValueError, match="Only mp4.*"):
            noio.VideoWriter(video_path, fps=30)

    def test_init_mp4(self, tmp_path: Path):
        video_path = tmp_path / "video.mp4"
        video_writer = noio.VideoWriter(video_path, fps=30)
        assert video_writer.where == video_path
        assert video_writer.fps == 30
        assert video_writer._container is None
        assert video_writer._stream is None


@pytest.fixture
def video_writer(tmp_path: Path) -> noio.VideoWriter:
    return noio.VideoWriter(tmp_path / "video.mp4", fps=30)


class TestVideoWriterPut:
    def test_put_array(self, img_array: np.ndarray, video_writer: noio.VideoWriter):
        video_writer.put(img_array)

        assert video_writer._container
        assert video_writer._stream

    def test_put_pil_img(self, img_pil: Image.Image, video_writer: noio.VideoWriter):
        video_writer.put(img_pil)

        assert video_writer._container
        assert video_writer._stream


class TestVideoWriterInitFunc:
    def test_init(self, video_writer: noio.VideoWriter):
        width, height = 100, 100
        video_writer.init_stream(width=width, height=height)

        assert video_writer._container
        assert video_writer._stream
        assert video_writer._stream.width == width
        assert video_writer._stream.height == height


@pytest.fixture
def used_video_writer(video_writer: noio.VideoWriter) -> noio.VideoWriter:
    video_writer._container = MagicMock()
    video_writer._stream = MagicMock(spec=VideoStream)
    return video_writer


class TestVideoWriterClose:
    def test_flush_and_close(self, used_video_writer: noio.VideoWriter):
        used_video_writer.close()
        assert used_video_writer._container
        used_video_writer._container.close.assert_called()
