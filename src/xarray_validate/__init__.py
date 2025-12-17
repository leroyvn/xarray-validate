"""
Main interface.
"""

from . import types as types
from ._version import version as __version__
from .base import (
    SchemaError,
    ValidationContext,
    ValidationMode,
    ValidationResult,
)
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
    "ArrayTypeSchema",
    "AttrSchema",
    "AttrsSchema",
    "ChunksSchema",
    "CoordsSchema",
    "DataArraySchema",
    "DatasetSchema",
    "DimsSchema",
    "DTypeSchema",
    "NameSchema",
    "SchemaError",
    "ShapeSchema",
    "testing",
    "types",
    "ValidationContext",
    "ValidationMode",
    "ValidationResult",
]
