from __future__ import annotations


from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Mapping,
)

import attrs as _attrs
import xarray as xr

from .base import BaseSchema, SchemaError
from .components import (
    ArrayTypeSchema,
    AttrsSchema,
    ChunksSchema,
    DimsSchema,
    DTypeSchema,
    NameSchema,
    ShapeSchema,
)


@_attrs.define(on_setattr=[_attrs.setters.convert, _attrs.setters.validate])
class CoordsSchema(BaseSchema):
    """
    Schema container for Coordinates

    Parameters
    ----------
    coords : dict
        Dict of coordinate keys and ``DataArraySchema`` objects

    require_all_keys : bool
        Whether require to all coordinates included in ``coords``

    allow_extra_keys : bool
        Whether to allow coordinates not included in ``coords`` dict

    Raises
    ------
    SchemaError
    """

    _json_schema = {
        "type": "object",
        "properties": {
            "require_all_keys": {
                "type": "boolean"
            },  # Question: is this the same as JSON's additionalProperties?
            "allow_extra_keys": {"type": "boolean"},
            "coords": {"type": "object"},
        },
    }

    coords: Dict[str, DataArraySchema] = _attrs.field()
    require_all_keys: bool = _attrs.field(default=True)
    allow_extra_keys: bool = _attrs.field(default=True)

    @classmethod
    def from_json(cls, obj: dict):
        coords = obj.pop("coords", {})
        coords = {k: DataArraySchema.from_json(v) for k, v in list(coords.items())}
        return cls(coords, **obj)

    def validate(self, coords: Mapping[str, Any]) -> None:
        """
        Validate coords

        Parameters
        ----------
        coords : dict_like
            coords of the DataArray. ``None`` may be used as a wildcard value
            for dict values.
        """

        if self.require_all_keys:
            missing_keys = set(self.coords) - set(coords)
            if missing_keys:
                raise SchemaError(f"coords has missing keys: {missing_keys}")

        if not self.allow_extra_keys:
            extra_keys = set(coords) - set(self.coords)
            if extra_keys:
                raise SchemaError(f"coords has extra keys: {extra_keys}")

        for key, da_schema in self.coords.items():
            if key not in coords:
                raise SchemaError(f"key {key} not in coords")
            else:
                da_schema.validate(coords[key])

    @property
    def json(self) -> dict:
        obj = {
            "require_all_keys": self.require_all_keys,
            "allow_extra_keys": self.allow_extra_keys,
            "coords": {k: v.json for k, v in self.coords.items()},
        }
        return obj


@_attrs.define(on_setattr=[_attrs.setters.convert, _attrs.setters.validate])
class DataArraySchema(BaseSchema):
    """
    A lightweight xarray.DataArray validator.

    Parameters
    ----------
    dtype : DTypeLike or str or DTypeSchema, optional
        Datatype of the the variable. If a string is specified, it must be a
        valid NumPy data type value.

    shape : ShapeT or tuple or ShapeSchema, optional
        Shape of the DataArray.

    dims : DimsT or list of str or DimsSchema, optional
        Dimensions of the DataArray.

    coords : CoordsSchema, optional
        Coordinates of the DataArray.

    chunks : bool or dict or ChunksSchema, optional
        If bool, specifies whether the DataArray is chunked or not, agnostic to
        chunk sizes. If dict, includes the expected chunks for the DataArray.

    name : str, optional
        Name of the DataArray.

    array_type : type, optional
        Type of the underlying data in a DataArray (*e.g.* :class:`numpy.ndarray`).

    checks : list of callables, optional
        List of callables that will further validate the DataArray.
    """

    _json_schema: ClassVar = {"type": "object"}
    _schema_slots: ClassVar = [
        "dtype",
        "dims",
        "shape",
        "coords",
        "name",
        "chunks",
        "attrs",
        "array_type",
    ]

    dtype: Optional[DTypeSchema] = _attrs.field(
        default=None,
        converter=_attrs.converters.optional(
            lambda x: x if isinstance(x, DTypeSchema) else DTypeSchema(x)
        ),
    )

    shape: Optional[ShapeSchema] = _attrs.field(
        default=None,
        converter=_attrs.converters.optional(
            lambda x: x if isinstance(x, ShapeSchema) else ShapeSchema(x)
        ),
    )

    dims: Optional[DimsSchema] = _attrs.field(
        default=None,
        converter=_attrs.converters.optional(
            lambda x: x if isinstance(x, DimsSchema) else DimsSchema(x)
        ),
    )

    name: Optional[NameSchema] = _attrs.field(
        default=None,
        converter=_attrs.converters.optional(
            lambda x: x if isinstance(x, NameSchema) else NameSchema(x)
        ),
    )

    coords: Optional[CoordsSchema] = _attrs.field(
        default=None,
        converter=_attrs.converters.optional(
            lambda x: x if isinstance(x, CoordsSchema) else CoordsSchema(x)
        ),
    )

    chunks: Optional[ChunksSchema] = _attrs.field(
        default=None,
        converter=_attrs.converters.optional(
            lambda x: x if isinstance(x, ChunksSchema) else ChunksSchema(x)
        ),
    )

    attrs: Optional[AttrsSchema] = _attrs.field(
        default=None,
        converter=_attrs.converters.optional(
            lambda x: x if isinstance(x, AttrsSchema) else AttrsSchema(x)
        ),
    )

    array_type: Optional[ArrayTypeSchema] = _attrs.field(
        default=None,
        converter=_attrs.converters.optional(
            lambda x: x if isinstance(x, ArrayTypeSchema) else ArrayTypeSchema(x)
        ),
    )

    checks: List[Callable] = _attrs.field(
        factory=list,
        validator=_attrs.validators.deep_iterable(_attrs.validators.is_callable()),
    )

    def validate(self, da: xr.DataArray) -> None:
        """
        Check if the DataArray complies with the Schema.

        Parameters
        ----------
        da : xr.DataArray
            DataArray to be validated

        Raises
        ------
        SchemaError
        """
        if not isinstance(da, xr.DataArray):
            raise ValueError("Input must be a xarray.DataArray")

        if self.dtype is not None:
            self.dtype.validate(da.dtype)

        if self.name is not None:
            self.name.validate(da.name)

        if self.dims is not None:
            self.dims.validate(da.dims)

        if self.shape is not None:
            self.shape.validate(da.shape)

        if self.coords is not None:
            self.coords.validate(da.coords)

        if self.chunks is not None:
            self.chunks.validate(da.chunks, da.dims, da.shape)

        if self.attrs:
            self.attrs.validate(da.attrs)

        if self.array_type is not None:
            self.array_type.validate(da.data)

        for check in self.checks:
            check(da)

    @property
    def json(self) -> dict:
        obj = {}
        for slot in self._schema_slots:
            try:
                obj[slot] = getattr(self, slot).json
            except AttributeError:
                pass
        return obj

    @classmethod
    def from_json(cls, obj: dict):
        kwargs = {}

        if "dtype" in obj:
            kwargs["dtype"] = DTypeSchema.from_json(obj["dtype"])
        if "shape" in obj:
            kwargs["shape"] = ShapeSchema.from_json(obj["shape"])
        if "dims" in obj:
            kwargs["dims"] = DimsSchema.from_json(obj["dims"])
        if "name" in obj:
            kwargs["name"] = NameSchema.from_json(obj["name"])
        if "coords" in obj:
            kwargs["coords"] = CoordsSchema.from_json(obj["coords"])
        if "chunks" in obj:
            kwargs["chunks"] = ChunksSchema.from_json(obj["chunks"])
        if "array_type" in obj:
            kwargs["array_type"] = ArrayTypeSchema.from_json(obj["array_type"])
        if "attrs" in obj:
            kwargs["attrs"] = AttrsSchema.from_json(obj["attrs"])

        return cls(**kwargs)
