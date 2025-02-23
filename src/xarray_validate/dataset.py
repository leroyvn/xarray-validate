from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, Union

import attrs as _attrs
import xarray as xr

from .base import BaseSchema, SchemaError
from .components import AttrsSchema
from .dataarray import CoordsSchema, DataArraySchema


@_attrs.define(on_setattr=[_attrs.setters.convert, _attrs.setters.validate])
class DatasetSchema(BaseSchema):
    r"""
    A lightweight xarray.Dataset validator.

    Parameters
    ----------
    data_vars : dict, optional
        Per-variable :class:`.DataArraySchema`\ s.

    coords : CoordsSchema, optional
        Coordinate validation schema.

    attrs : AttrsSchema, optional
        Attributes value validation schema.

    checks : list of callables, optional
        List of callables that will further validate the Dataset.
    """

    data_vars: Optional[Dict[str, Optional[DataArraySchema]]] = _attrs.field(
        default=None,
        converter=_attrs.converters.optional(
            lambda x: {
                k: v if isinstance(v, DataArraySchema) else DataArraySchema(**v)
                for k, v in x.items()
            }
        ),
    )

    coords: Union[CoordsSchema, None] = _attrs.field(
        default=None, converter=_attrs.converters.optional(CoordsSchema.convert)
    )

    attrs: Union[AttrsSchema, None] = _attrs.field(
        default=None,
        converter=_attrs.converters.optional(
            lambda x: x if isinstance(x, AttrsSchema) else AttrsSchema(x)
        ),
    )

    checks: Iterable[Callable] = _attrs.field(
        factory=list,
        validator=_attrs.validators.deep_iterable(_attrs.validators.is_callable()),
    )

    def serialize(self):
        obj = {
            "data_vars": {},
            "attrs": self.attrs.serialize() if self.attrs is not None else {},
        }
        if self.data_vars:
            for key, var in self.data_vars.items():
                obj["data_vars"][key] = var.serialize()
        if self.coords:
            obj["coords"] = self.coords.serialize()
        return obj

    @classmethod
    def deserialize(cls, obj: dict):
        kwargs = {}

        if "data_vars" in obj:
            kwargs["data_vars"] = {
                k: DataArraySchema.convert(v) for k, v in obj["data_vars"].items()
            }
        if "coords" in obj:
            kwargs["coords"] = CoordsSchema.convert(obj["coords"])
        if "attrs" in obj:
            kwargs["attrs"] = AttrsSchema.convert(obj["attrs"])

        return cls(**kwargs)

    @classmethod
    def from_dataset(cls, value):
        ds_schema = value.to_dict(data=False)
        return cls.deserialize(ds_schema)

    def validate(self, ds: xr.Dataset) -> None:
        # Inherit docstring
        
        if self.data_vars is not None:
            for key, da_schema in self.data_vars.items():
                if da_schema is not None:
                    if key not in ds.data_vars:
                        raise SchemaError(f"data variable {key} not in ds")
                    else:
                        da_schema.validate(ds.data_vars[key])

        if self.coords is not None:  # pragma: no cover
            self.coords.validate(ds.coords)

        if self.attrs:
            self.attrs.validate(ds.attrs)

        if self.checks:
            for check in self.checks:
                check(ds)
