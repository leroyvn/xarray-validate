from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Dict, Hashable, Mapping, Optional, Tuple, Union, ClassVar

import attrs as _attrs
import numpy as np
from numpy import dtype

from .base import BaseSchema, SchemaError
from .types import ChunksT, DimsT, DTypeLike, ShapeT


@_attrs.define(on_setattr=[_attrs.setters.convert, _attrs.setters.validate])
class DTypeSchema(BaseSchema):
    """
    Datatype schema

    Parameters
    ----------
    dtype : DTypeLike
        Datatype definition, may be (string, np.dtype, etc.)

    Raises
    ------
    SchemaError
    """

    _json_schema: ClassVar = {"type": "string"}

    dtype: dtype = _attrs.field(
        converter=lambda x: x
        if x
        in [
            np.floating,
            np.integer,
            np.signedinteger,
            np.unsignedinteger,
            np.generic,
        ]
        else np.dtype(x)
    )

    @classmethod
    def from_json(cls, obj: str):
        if obj in [
            "floating",
            "integer",
            "signedinteger",
            "unsignedinteger",
            "generic",
        ]:
            dtype = getattr(np, obj)
        else:
            dtype = obj
        return cls(dtype)

    def validate(self, dtype: DTypeLike) -> None:
        """
        Validate dtype

        Parameters
        ----------
        dtype : Any
            Dtype of the DataArray.
        """
        if not np.issubdtype(dtype, self.dtype):
            raise SchemaError(f"dtype {dtype} != {self.dtype}")

    @property
    def json(self) -> str:
        if isinstance(self.dtype, np.dtype):
            return self.dtype.str
        else:
            # fallbacks
            return str(getattr(self.dtype, "__name__", str(self.dtype)))


@_attrs.define(on_setattr=[_attrs.setters.convert, _attrs.setters.validate])
class DimsSchema(BaseSchema):
    """
    Dimensions schema

    Parameters
    ----------
    dims : str or iterable of str
        Dimensions definition, ``None`` may be used as a wildcard.

    Raises
    ------
    SchemaError
    """

    _json_schema: ClassVar = {"type": "array", "items": {"type": ["string", "null"]}}

    dims: DimsT = _attrs.field()

    @classmethod
    def from_json(cls, obj: DimsT):
        return cls(obj)

    def validate(self, dims: tuple) -> None:
        """
        Validate dimensions

        Parameters
        ----------
        dims : Tuple[Union[str, None]]
            Dimensions of the DataArray. `None` may be used as a wildcard value.
        """
        if len(self.dims) != len(dims):
            raise SchemaError(
                f"length of dims does not match: {len(dims)} != {len(self.dims)}"
            )

        for i, (actual, expected) in enumerate(zip(dims, self.dims)):
            if expected is not None and actual != expected:
                raise SchemaError(f"dim mismatch in axis {i}: {actual} != {expected}")

    @property
    def json(self) -> list:
        return list(self.dims)


@_attrs.define(on_setattr=[_attrs.setters.convert, _attrs.setters.validate])
class ShapeSchema(BaseSchema):
    """
    Shape schema

    Parameters
    ----------
    shape : iterable of ints
        Shape definition, ``None`` may be used as a wildcard.

    Raises
    ------
    SchemaError
    """

    _json_schema: ClassVar = {"type": "array"}

    shape: ShapeT = _attrs.field()

    @classmethod
    def from_json(cls, obj: ShapeT):
        return cls(obj)

    def validate(self, shape: tuple) -> None:
        """
        Validate shape

        Parameters
        ----------
        shape : ShapeT
            Shape of the DataArray. `None` may be used as a wildcard value.
        """
        if len(self.shape) != len(shape):
            raise SchemaError(
                f"number of dimensions in shape ({len(shape)}) != da.ndim ({len(self.shape)})"
            )

        for i, (actual, expected) in enumerate(zip(shape, self.shape)):
            if expected is not None and actual != expected:
                raise SchemaError(f"shape mismatch in axis {i}: {actual} != {expected}")

    @property
    def json(self) -> list:
        return list(self.shape)


@_attrs.define(on_setattr=[_attrs.setters.convert, _attrs.setters.validate])
class NameSchema(BaseSchema):
    """
    Name schema

    Parameters
    ----------
    name : str
        Name definition.

    Raises
    ------
    SchemaError
    """

    _json_schema: ClassVar = {"type": "string"}

    name: str = _attrs.field()

    @classmethod
    def from_json(cls, obj: str):
        return cls(obj)

    def validate(self, name: Hashable) -> None:
        """
        Validate name

        Parameters
        ----------
        name : str, optional
            Name of the DataArray. Currently requires an exact string match.
        """
        # TODO: support regular expressions
        # - http://json-schema.org/understanding-json-schema/reference/regular_expressions.html
        # - https://docs.python.org/3.9/library/re.html
        if self.name != name:
            raise SchemaError(f"name {name} != {self.name}")

    @property
    def json(self) -> str:
        return self.name


@_attrs.define(on_setattr=[_attrs.setters.convert, _attrs.setters.validate])
class ChunksSchema(BaseSchema):
    """
    Chunks schema

    Parameters
    ----------
    chunks : dict or bool
        Chunks definition. If ``bool``, whether validated object should be chunked.
        If ``dict``, mapping of dimension name to chunk size. None may be used as a wildcard.

    Raises
    ------
    SchemaError
    """

    _json_schema: ClassVar = {"type": ["boolean", "object"]}

    chunks: ChunksT = _attrs.field()

    @classmethod
    def from_json(cls, obj: dict):
        return cls(obj)  # TODO: this will likely need input validation.

    def validate(
        self,
        chunks: Optional[Tuple[Tuple[int, ...], ...]],
        dims: Tuple,
        shape: Tuple[int, ...],
    ) -> None:
        """
        Validate chunks

        Parameters
        ----------
        chunks : tuple
            Chunks from ``DataArray.chunks``
        dims : tuple of str
            Dimension keys from array.
        shape : tuple of int
            Shape of array.
        """

        if isinstance(self.chunks, bool):
            if self.chunks and not chunks:
                raise SchemaError("expected array to be chunked but it is not")
            elif not self.chunks and chunks:
                raise SchemaError("expected unchunked array but it is chunked")
        elif isinstance(self.chunks, dict):
            if chunks is None:
                raise SchemaError("expected array to be chunked but it is not")
            dim_chunks = dict(zip(dims, chunks))
            dim_sizes = dict(zip(dims, shape))
            # check whether chunk sizes are regular because we assume the first chunk to be representative below
            for key, ec in self.chunks.items():
                if isinstance(ec, int):
                    # handles case of expected chunksize is shorthand of -1 which translates to the full length of dimension
                    if ec < 0:
                        ec = dim_sizes[key]
                    ac = dim_chunks[key]
                    if any([a != ec for a in ac[:-1]]) or ac[-1] > ec:
                        raise SchemaError(f"{key} chunks did not match: {ac} != {ec}")

                else:  # assumes ec is an iterable
                    ac = dim_chunks[key]
                    if ec is not None and tuple(ac) != tuple(ec):
                        raise SchemaError(f"{key} chunks did not match: {ac} != {ec}")
        else:
            raise ValueError(f"got unknown chunks type: {type(self.chunks)}")

    @property
    def json(self) -> Union[bool, Dict[str, Any]]:
        if isinstance(self.chunks, bool):
            return self.chunks
        else:
            obj = {}
            for key, val in self.chunks.items():
                if isinstance(val, Iterable):
                    obj[key] = list(val)
                else:
                    obj[key] = val
            return obj


@_attrs.define(on_setattr=[_attrs.setters.convert, _attrs.setters.validate])
class ArrayTypeSchema(BaseSchema):
    """
    Array type schema

    Parameters
    ----------
    array_type : str or object
        Array type definition.

    Raises
    ------
    SchemaError
    """

    _json_schema: ClassVar = {"type": "string"}

    array_type = _attrs.field()

    @classmethod
    def from_json(cls, obj: str):
        array_type: (
            Any  # TODO: figure out how to optionally include the dask array type
        )

        if obj == "<class 'dask.array.core.Array'>":
            import dask.array as da

            array_type = da.Array
        elif obj == "<class 'numpy.ndarray'>":
            array_type = np.ndarray
        else:
            raise ValueError(f"unknown array_type: {obj}")
        return cls(array_type)

    def validate(self, array: Any) -> None:
        """
        Validate array_type

        Parameters
        ----------
        array : array_like
            array_type of the DataArray. `None` may be used as a wildcard value.
        """
        if not isinstance(array, self.array_type):
            raise SchemaError(f"array_type {type(array)} != {self.array_type}")

    @property
    def json(self) -> str:
        return str(self.array_type)


@_attrs.define(on_setattr=[_attrs.setters.convert, _attrs.setters.validate])
class AttrSchema(BaseSchema):
    """
    Attribute schema

    Parameters
    ----------
    type : object
        Attribute type definition.
    value :
        Attribute value definition.

    Raises
    ------
    SchemaError
    """

    _json_schema: ClassVar = {
        "type": "string",
        "value": ["string", "number", "array", "boolean", "null"],
    }

    type: Optional[str] = _attrs.field(default=None)
    value: Optional[Any] = _attrs.field(default=None)

    @classmethod
    def from_json(cls, obj: str):
        return cls(obj)

    def validate(self, attr: Any):
        """
        Validate attrs

        Parameters
        ----------
        attr : any
            attribute, `None` may be used as a wildcard value.
        """
        if self.type is not None:
            if not isinstance(attr, self.type):
                raise SchemaError(f"attrs {attr} is not of type {self.type}")

        if self.value is not None:
            if self.value is not None and self.value != attr:
                raise SchemaError(f"name {attr} != {self.value}")

    @property
    def json(self) -> dict:
        return {"type": self.type, "value": self.value}


@_attrs.define(on_setattr=[_attrs.setters.convert, _attrs.setters.validate])
class AttrsSchema(BaseSchema):
    """
    Attributes schema

    Parameters
    ----------
    attrs : str or iterable of str
        Attributes definition
    require_all_keys : bool
        Whether require to all coordinates included in ``attrs``
    allow_extra_keys : bool
        Whether to allow coordinates not included in ``attrs`` dict

    Raises
    ------
    SchemaError
    """

    _json_schema: ClassVar = {
        "type": "object",
        "properties": {
            "require_all_keys": {
                "type": "boolean"
            },  # Question: is this the same as JSON's additionalProperties?
            "allow_extra_keys": {"type": "boolean"},
            "attrs": {"type": "object"},
        },
    }

    attrs: Mapping[Hashable, AttrSchema] = _attrs.field()
    require_all_keys: bool = _attrs.field(default=True)
    allow_extra_keys: bool = _attrs.field(default=True)

    @classmethod
    def from_json(cls, obj: dict):
        attrs = {}
        if "attrs" not in obj:
            raise ValueError("missing 'attrs' field")

        for key, val in obj["attrs"].items():
            if val is None:
                val = {}
            attrs[key] = AttrSchema(**val)

        kwargs = {}

        if "require_all_keys" in obj:
            kwargs["require_all_keys"] = obj["require_all_keys"]
        if "allow_extra_keys" in obj:
            kwargs["allow_extra_keys"] = obj["allow_extra_keys"]

        return cls(attrs, **kwargs)

    def validate(self, attrs: Any) -> None:
        """
        Validate attrs

        Parameters
        ----------
        attrs : dict_like
            attrs dict, `None` may be used as a wildcard value.
        """

        if self.require_all_keys:
            missing_keys = set(self.attrs) - set(attrs)
            if missing_keys:
                raise SchemaError(f"attrs has missing keys: {missing_keys}")

        if not self.allow_extra_keys:
            extra_keys = set(attrs) - set(self.attrs)
            if extra_keys:
                raise SchemaError(f"attrs has extra keys: {extra_keys}")

        for key, attr_schema in self.attrs.items():
            if key not in attrs:
                raise SchemaError(f"key {key} not in attrs")
            else:
                attr_schema.validate(attrs[key])

    @property
    def json(self) -> dict:
        obj = {
            "require_all_keys": self.require_all_keys,
            "allow_extra_keys": self.allow_extra_keys,
            "attrs": {k: v.json for k, v in self.attrs.items()},
        }
        return obj
