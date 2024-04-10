"""Handling of inputs/outputs."""

from pathlib import Path
from typing import Any, Union

import av
import numpy as np
from PIL import Image

from niftyone.image._convert import topil
from niftyone.typing import StrPath


class VideoWriter:
    """A simple video streaming writer."""

    def __init__(self, where: StrPath, fps: int) -> None:
        """Initialize class."""
        where = Path(where)
        if where.suffix != ".mp4":
            raise ValueError("Only mp4 output supported")

        self.where = where
        self.fps = fps
        self._container = None
        self._stream = None

    def put(self, img: Union[np.ndarray, Image.Image]) -> None:
        """Add frame to the stream."""
        if isinstance(img, np.ndarray):
            img = topil(img)

        if self._container is None:
            self.init_stream(width=img.width, height=img.height)

        frame = av.VideoFrame.from_image(img)
        for packet in self._stream.encode(frame):  # type: ignore [attr-defined]
            self._container.mux(packet)  # type: ignore [attr-defined]

    def init_stream(self, width: int, height: int) -> None:
        """Initialize the stream."""
        self._container = av.open(str(self.where), mode="w")
        self._stream = self._container.add_stream("h264", rate=self.fps)  # type: ignore [attr-defined]
        self._stream.width = width  # type: ignore [attr-defined]
        self._stream.height = height  # type: ignore [attr-defined]
        self._stream.pix_fmt = "yuv420p"  # type: ignore [attr-defined]

    def close(self) -> None:
        """Close the stream."""
        if self._container is not None:
            # Flush stream
            for packet in self._stream.encode():
                self._container.mux(packet)
            # Close the file
            self._container.close()

    def __enter__(self) -> "VideoWriter":
        """Create and return class, allows for context management."""
        return self

    def __exit__(self, *args: tuple[Any]) -> None:
        """Ensure class is closed."""
        self.close()
