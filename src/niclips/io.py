"""Handling of inputs/outputs."""

import logging
from pathlib import Path
from typing import Any

import av
import nibabel as nib
import numpy as np
from av.container import OutputContainer
from av.stream import Stream
from av.video.stream import VideoStream
from PIL import Image

from niclips.image._convert import topil
from niclips.typing import StrPath

try:
    import nifti

    HAVE_NIFTI = True
except ImportError:  # pragma: no cover
    HAVE_NIFTI = False


def load_nifti(fpath: str | Path, use_niftilib: bool = True) -> nib.Nifti1Image:
    """Wrapper to load Nifti images using library."""
    # Uses nifti library if available and user selects it
    use_niftilib = use_niftilib and HAVE_NIFTI
    if use_niftilib:
        hdr, arr = nifti.read_volume(str(fpath))
        new_hdr = nib.Nifti1Header()
        for key, val in hdr.items():
            if key in new_hdr:
                new_hdr[key] = val
        aff = new_hdr.get_best_affine()
        return nib.Nifti1Image(dataobj=arr, affine=aff, header=new_hdr)
    else:
        if not HAVE_NIFTI:
            logging.warning("`nifti` library is unavailable - using `nibabel`")
        return nib.load(fpath)


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

    def put(self, img: np.ndarray | Image.Image) -> None:
        """Add frame to the stream."""
        if isinstance(img, np.ndarray):
            img = topil(img)

        if self._container is None:
            self.init_stream(width=img.width, height=img.height)

        frame = av.VideoFrame.from_image(img)
        assert isinstance(self._stream, VideoStream)
        for packet in self._stream.encode(frame):
            assert self._container
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
        if self._container:
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
