"""Defined types used in niclips."""

import os
import typing
from types import UnionType

import nibabel as nib
import numpy as np

StrPath = str | os.PathLike

Coord = tuple[float, float, float] | np.ndarray

NiftiLike = nib.nifti1.Nifti1Image | np.ndarray


def get_union_subclass(
    union_type: typing.Type[typing.Any], subclass: typing.Type[typing.Any]
) -> bool:
    """Function to check for union types."""
    origin = typing.get_origin(union_type)
    if origin is UnionType:
        return any(
            issubclass(union_cls, subclass) for union_cls in typing.get_args(union_type)
        )
    return issubclass(union_type, subclass)
