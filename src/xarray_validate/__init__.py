"""
Main interface.
"""

from . import types as types
from ._version import version as __version__
from .base import SchemaError
from .components import (
    ArrayTypeSchema,
    AttrSchema,
    AttrsSchema,
    ChunksSchema,
    DimsSchema,
    DTypeSchema,
    NameSchema,
    ShapeSchema,
)
from .dataarray import CoordsSchema, DataArraySchema
from .dataset import DatasetSchema

__all__ = [
    "__version__",
    "SchemaError",
    "ArrayTypeSchema",
    "AttrSchema",
    "AttrsSchema",
    "ChunksSchema",
    "DimsSchema",
    "DTypeSchema",
    "NameSchema",
    "ShapeSchema",
    "CoordsSchema",
    "DataArraySchema",
    "DatasetSchema",
    "types",
]
