"""
Main interface.
"""

from .base import SchemaError
from .components import ArrayTypeSchema
from .components import AttrSchema
from .components import AttrsSchema
from .components import ChunksSchema
from .components import DimsSchema
from .components import DTypeSchema
from .components import NameSchema
from .components import ShapeSchema
from .dataarray import CoordsSchema
from .dataarray import DataArraySchema
from .dataset import DatasetSchema

__all__ = [
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
]
