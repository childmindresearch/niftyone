"""Handling of inputs/outputs."""

from pathlib import Path
from typing import Any, Union

import av
import numpy as np
from av.container import OutputContainer
from av.stream import Stream
from av.video.stream import VideoStream
from PIL import Image

from niclips.image._convert import topil
from niclips.typing import StrPath


class VideoWriter:
    """A simple video streaming writer."""

    def __init__(self, where: StrPath, fps: int) -> None:
        where = Path(where)
        if where.suffix != ".mp4":
            raise ValueError("Only mp4 output supported")

        self.where = where
        self.fps = fps
        self._container: OutputContainer | None = None
        self._stream: Stream | VideoStream | None = None

    def put(self, img: Union[np.ndarray, Image.Image]) -> None:
        """Add frame to the stream."""
        if isinstance(img, np.ndarray):
            img = topil(img)

        if self._container is None:
            self.init_stream(width=img.width, height=img.height)

        frame = av.VideoFrame.from_image(img)
        assert isinstance(self._stream, VideoStream)
        for packet in self._stream.encode(frame):
            assert self._container is not None
            self._container.mux_one(packet)

    def init_stream(self, width: int, height: int) -> None:
        """Initialize the stream."""
        self._container = av.open(str(self.where), mode="w")
        self._stream = self._container.add_stream("h264", rate=self.fps)
        assert isinstance(self._stream, VideoStream)
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = "yuv420p"

    def close(self) -> None:
        """Close the stream."""
        if self._container is not None:
            # Flush stream
            assert isinstance(self._stream, VideoStream)
            for packet in self._stream.encode():
                self._container.mux_one(packet)
            # Close the file
            self._container.close()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *args: tuple[Any]) -> None:
        self.close()
