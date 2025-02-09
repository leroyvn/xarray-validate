from typing import Callable, ClassVar, Dict, Iterable, Optional, Union

import attrs as _attrs
import xarray as xr

from .base import BaseSchema, SchemaError
from .components import AttrSchema, AttrsSchema
from .dataarray import CoordsSchema, DataArraySchema


@_attrs.define(on_setattr=[_attrs.setters.convert, _attrs.setters.validate])
class DatasetSchema(BaseSchema):
    r"""
    A lightweight xarray.Dataset validator.

    Parameters
    ----------
    data_vars : dict, optional
        Per-variable :class:`.DataArraySchema`\ s.

    checks : list of callables, optional
        List of callables that will further validate the Dataset.
    """

    _json_schema: ClassVar = {
        "type": "object",
        "properties": {
            "data_vars": {"type": "object"},
            "coords": {"type": "object"},
            "attrs": {"type": "object"},
        },
    }

    data_vars: Optional[Dict[str, Optional[DataArraySchema]]] = _attrs.field(
        default=None,
        converter=_attrs.converters.optional(
            lambda x: {
                k: v if isinstance(v, DataArraySchema) else DataArraySchema(**v)
                for k, v in x.items()
            }
        ),
    )

    coords: Union[CoordsSchema, Dict[str, DataArraySchema], None] = _attrs.field(
        default=None
    )

    attrs: Union[AttrsSchema, Dict[str, AttrSchema], None] = _attrs.field(
        default=None,
        converter=_attrs.converters.optional(
            lambda x: x if isinstance(x, AttrsSchema) else AttrsSchema(x)
        ),
    )

    checks: Iterable[Callable] = _attrs.field(
        factory=list,
        validator=_attrs.validators.deep_iterable(_attrs.validators.is_callable()),
    )

    @classmethod
    def from_json(cls, obj: dict):
        kwargs = {}
        if "data_vars" in obj:
            kwargs["data_vars"] = {
                k: DataArraySchema.from_json(v) for k, v in obj["data_vars"].items()
            }
        if "coords" in obj:
            kwargs["coords"] = {
                k: CoordsSchema.from_json(v) for k, v in obj["coords"].items()
            }
        if "attrs" in obj:
            kwargs["attrs"] = {
                k: AttrsSchema.from_json(v) for k, v in obj["attrs"].items()
            }

        return cls(**kwargs)

    def validate(self, ds: xr.Dataset) -> None:
        """
        Check if the Dataset complies with the Schema.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset to be validated

        Returns
        -------
        xr.Dataset
            Validated Dataset

        Raises
        ------
        SchemaError
        """

        if self.data_vars is not None:
            for key, da_schema in self.data_vars.items():
                if da_schema is not None:
                    if key not in ds.data_vars:
                        raise SchemaError(f"data variable {key} not in ds")
                    else:
                        da_schema.validate(ds.data_vars[key])

        if self.coords is not None:  # pragma: no cover
            raise NotImplementedError("coords schema not implemented yet")

        if self.attrs:
            self.attrs.validate(ds.attrs)

        if self.checks:
            for check in self.checks:
                check(ds)

    @property
    def json(self):
        obj = {
            "data_vars": {},
            "attrs": self.attrs.json if self.attrs is not None else {},
        }
        if self.data_vars:
            for key, var in self.data_vars.items():
                obj["data_vars"][key] = var.json
        if self.coords:
            obj["coords"] = self.coords.json
        return obj
