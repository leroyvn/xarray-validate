from __future__ import annotations

from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Mapping,
    Optional,
)

import attrs as _attrs
import numpy as np
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
        Whether to require to all coordinates included in ``coords``

    allow_extra_keys : bool
        Whether to allow coordinates not included in ``coords`` dict
    """

    coords: Dict[str, DataArraySchema] = _attrs.field()
    require_all_keys: bool = _attrs.field(default=True)
    allow_extra_keys: bool = _attrs.field(default=True)

    def serialize(self) -> dict:
        obj = {
            "require_all_keys": self.require_all_keys,
            "allow_extra_keys": self.allow_extra_keys,
            "coords": {k: v.serialize() for k, v in self.coords.items()},
        }
        return obj

    @classmethod
    def deserialize(cls, obj: dict):
        if "coords" in obj:
            coords = obj["coords"]
            kwargs = {k: v for k, v in obj.items() if k != "coords"}
        else:
            coords = obj
            kwargs = {}

        coords = {k: DataArraySchema.convert(v) for k, v in list(coords.items())}
        return cls(coords=coords, **kwargs)

    def validate(self, coords: Mapping[str, Any]) -> None:
        # Inherit docstring

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


@_attrs.define(on_setattr=[_attrs.setters.convert, _attrs.setters.validate])
class DataArraySchema(BaseSchema):
    """
    A lightweight xarray.DataArray validator.

    Parameters
    ----------
    dtype : DTypeLike or str or DTypeSchema, optional
        Data type validation schema. If a string is specified, it must be a
        valid NumPy data type value.

    shape : ShapeT or tuple or ShapeSchema, optional
        Shape validation schema.

    dims : DimsT or list of str or DimsSchema, optional
        Dimensions validation schema.

    name : str, optional
        Name validation schema.

    coords : CoordsSchema, optional
        Coordinates validation schema.

    chunks : bool or dict or ChunksSchema, optional
        If bool, specifies whether the DataArray is chunked or not, agnostic to
        chunk sizes. If dict, includes the expected chunks for the DataArray.

    attrs : AttrsSchema, optional
        Attributes validation schema.

    array_type : type, optional
        Type of the underlying data in a DataArray (*e.g.* :class:`numpy.ndarray`).

    checks : list of callables, optional
        List of callables that will further validate the DataArray.
    """

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

    dtype: np.dtype = _attrs.field(
        default=None,
        converter=_attrs.converters.optional(DTypeSchema.convert),
    )

    shape: Optional[ShapeSchema] = _attrs.field(
        default=None,
        converter=_attrs.converters.optional(ShapeSchema.convert),
    )

    dims: Optional[DimsSchema] = _attrs.field(
        default=None,
        converter=_attrs.converters.optional(DimsSchema.convert),
    )

    name: Optional[NameSchema] = _attrs.field(
        default=None,
        converter=_attrs.converters.optional(NameSchema.convert),
    )

    coords: Optional[CoordsSchema] = _attrs.field(
        default=None,
        converter=_attrs.converters.optional(CoordsSchema.convert),
    )

    chunks: Optional[ChunksSchema] = _attrs.field(
        default=None,
        converter=_attrs.converters.optional(ChunksSchema.convert),
    )

    attrs: Optional[AttrsSchema] = _attrs.field(
        default=None,
        converter=_attrs.converters.optional(AttrsSchema.convert),
    )

    array_type: Optional[ArrayTypeSchema] = _attrs.field(
        default=None,
        converter=_attrs.converters.optional(ArrayTypeSchema.convert),
    )

    checks: List[Callable] = _attrs.field(
        factory=list,
        validator=_attrs.validators.deep_iterable(_attrs.validators.is_callable()),
    )

    def serialize(self) -> dict:
        obj = {}
        for slot in self._schema_slots:
            try:
                obj[slot] = getattr(self, slot).serialize()
            except AttributeError:
                pass
        return obj

    @classmethod
    def deserialize(cls, obj: dict):
        kwargs = {}

        if "dtype" in obj:
            kwargs["dtype"] = DTypeSchema.convert(obj["dtype"])
        if "shape" in obj:
            kwargs["shape"] = ShapeSchema.convert(obj["shape"])
        if "dims" in obj:
            kwargs["dims"] = DimsSchema.convert(obj["dims"])
        if "name" in obj:
            kwargs["name"] = NameSchema.convert(obj["name"])
        if "coords" in obj:
            kwargs["coords"] = CoordsSchema.convert(obj["coords"])
        if "chunks" in obj:
            kwargs["chunks"] = ChunksSchema.convert(obj["chunks"])
        if "array_type" in obj:
            kwargs["array_type"] = ArrayTypeSchema.convert(obj["array_type"])
        if "attrs" in obj:
            kwargs["attrs"] = AttrsSchema.convert(obj["attrs"])

        return cls(**kwargs)

    @classmethod
    def from_dataarray(cls, value: xr.DataArray):
        da_schema = value.to_dict(data=False)
        da_schema["coords"] = {"coords": da_schema["coords"]}
        da_schema["attrs"] = {"attrs": da_schema["attrs"]}
        return cls.deserialize(da_schema)

    def validate(self, da: xr.DataArray) -> None:
        # Inherit docstring

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
