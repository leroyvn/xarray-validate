from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Dict, Hashable, Optional, Tuple, Union

import attrs as _attrs
import numpy as np
from numpy import dtype
from numpy.typing import DTypeLike

from . import converters
from .base import BaseSchema, SchemaError
from .types import ChunksT, DimsT, ShapeT


@_attrs.define(on_setattr=[_attrs.setters.convert, _attrs.setters.validate])
class DTypeSchema(BaseSchema):
    """
    Datatype schema.

    Parameters
    ----------
    dtype : DTypeLike
        DataArray dtype.
    """

    dtype: dtype = _attrs.field(converter=np.dtype)

    def serialize(self):
        # Inherit docstring
        return self.dtype.str

    @classmethod
    def deserialize(cls, obj):
        # Inherit docstring
        return cls(obj)

    def validate(self, dtype: DTypeLike) -> None:
        # Inherit docstring

        if not np.issubdtype(dtype, self.dtype):
            raise SchemaError(
                f"dtype mismatch: got {repr(dtype)}, expected {repr(self.dtype)}"
            )


@_attrs.define(on_setattr=[_attrs.setters.convert, _attrs.setters.validate])
class DimsSchema(BaseSchema):
    """
    Dimensions schema.

    Parameters
    ----------
    dims : sequence of (str or None)
        DataArray dimensions. ``None`` may be used as a wildcard.
    """

    dims: DimsT = _attrs.field(
        converter=lambda x: tuple(x) if not isinstance(x, str) else x,
        validator=_attrs.validators.deep_iterable(
            member_validator=_attrs.validators.optional(
                _attrs.validators.instance_of(str)
            )
        ),
    )

    def serialize(self) -> list:
        # Inherit docstring
        return list(self.dims)

    @classmethod
    def deserialize(cls, obj: DimsT) -> DimsSchema:
        # Inherit docstring
        return cls(obj)

    def validate(self, dims: DimsT) -> None:
        # Inherit docstring

        if len(self.dims) != len(dims):
            raise SchemaError(
                f"dimension number mismatch: got {len(dims)}, expected {len(self.dims)}"
            )

        for i, (actual, expected) in enumerate(zip(dims, self.dims)):
            if expected is not None and actual != expected:
                raise SchemaError(
                    f"dimension mismatch in axis {i}: got {actual}, expected {expected}"
                )


@_attrs.define(on_setattr=[_attrs.setters.convert, _attrs.setters.validate])
class ShapeSchema(BaseSchema):
    """
    Shape schema.

    Parameters
    ----------
    shape : sequence of (int or None)
        Shape of the DataArray. ``None`` may be used as a wildcard.
    """

    shape: ShapeT = _attrs.field(
        converter=lambda x: tuple(x) if not isinstance(x, int) else x,
        validator=_attrs.validators.deep_iterable(
            member_validator=_attrs.validators.optional(
                _attrs.validators.instance_of(int)
            )
        ),
    )

    def serialize(self) -> list:
        # Inherit docstring
        return list(self.shape)

    @classmethod
    def deserialize(cls, obj: ShapeT):
        # Inherit docstring
        return cls(obj)

    def validate(self, shape: tuple) -> None:
        # Inherit docstring

        if len(self.shape) != len(shape):
            raise SchemaError(
                "dimension count mismatch: "
                f"got {len(shape)}, expected {len(self.shape)}"
            )

        for i, (actual, expected) in enumerate(zip(shape, self.shape)):
            if expected is not None and actual != expected:
                raise SchemaError(
                    f"shape mismatch in axis {i}: got {actual}, expected {expected}"
                )


@_attrs.define(on_setattr=[_attrs.setters.convert, _attrs.setters.validate])
class NameSchema(BaseSchema):
    """
    Name schema.

    Parameters
    ----------
    name : str
        Name definition.
    """

    name: str = _attrs.field(converter=str)

    def serialize(self) -> str:
        # Inherit docstring
        return self.name

    @classmethod
    def deserialize(cls, obj: str):
        # Inherit docstring
        return cls(obj)

    def validate(self, name: Hashable) -> None:
        # Inherit docstring

        # TODO: support regular expressions
        # - http://json-schema.org/understanding-json-schema/reference/regular_expressions.html
        # - https://docs.python.org/3.9/library/re.html
        if self.name != name:
            raise SchemaError(f"name mismatch: got {name}, expected {self.name}")


@_attrs.define(on_setattr=[_attrs.setters.convert, _attrs.setters.validate])
class ChunksSchema(BaseSchema):
    """
    Chunks schema.

    Parameters
    ----------
    chunks : dict or bool
        Chunks definition. If ``bool``, whether the validated object should be
        chunked. If ``dict``, mapping of dimension name to chunk size. ``None``
        may be used as a wildcard.
    """

    chunks: ChunksT = _attrs.field(
        validator=_attrs.validators.instance_of((bool, dict))
    )

    def serialize(self) -> Union[bool, Dict[str, Any]]:
        # Inherit docstring
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

    @classmethod
    def deserialize(cls, obj: dict):
        # Inherit docstring
        return cls(obj)

    def validate(
        self,
        chunks: Optional[Tuple[Tuple[int, ...], ...]],
        dims: Tuple,
        shape: Tuple[int, ...],
    ) -> None:
        """
        Validate chunks against this schema.

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
            # Check whether chunk sizes are regular because we assume the first
            # chunk to be representative below
            for key, ec in self.chunks.items():
                if isinstance(ec, int):
                    # Handles case of expected chunk size is shorthand of -1 which
                    # translates to the full length of dimension
                    if ec < 0:
                        ec = dim_sizes[key]
                    ac = dim_chunks[key]
                    if any([a != ec for a in ac[:-1]]) or ac[-1] > ec:
                        raise SchemaError(
                            f"chunk mismatch for {key}: got {ac}, expected {ec}"
                        )

                else:  # assumes ec is an iterable
                    ac = dim_chunks[key]
                    if ec is not None and tuple(ac) != tuple(ec):
                        raise SchemaError(
                            f"chunk mismatch for {key}: got {ac}, expected {ec}"
                        )
        else:
            raise ValueError(f"got unknown chunks type: {type(self.chunks)}")


@_attrs.define(on_setattr=[_attrs.setters.convert, _attrs.setters.validate])
class ArrayTypeSchema(BaseSchema):
    """
    Array type schema.

    Parameters
    ----------
    array_type : str or type
        Array type definition.
    """

    array_type: type = _attrs.field(
        converter=converters.array_type_converter,
        validator=_attrs.validators.instance_of(type),
    )

    def serialize(self) -> str:
        # Inherit docstring
        return str(self.array_type)

    @classmethod
    def deserialize(cls, obj: str):
        return cls(obj)

    def validate(self, array: Any) -> None:
        # Inherit docstring

        if not isinstance(array, self.array_type):
            raise SchemaError(
                f"array type mismatch: got {type(array)}, expected {self.array_type}"
            )


@_attrs.define(on_setattr=[_attrs.setters.convert, _attrs.setters.validate])
class AttrSchema(BaseSchema):
    """
    Attribute schema.

    Parameters
    ----------
    type : type, optional
        Attribute type definition. ``None`` may be used as a wildcard.

    value : Any
        Attribute value definition. ``None`` may be used as a wildcard.
    """

    type: Optional[str] = _attrs.field(
        default=None,
        validator=_attrs.validators.optional(_attrs.validators.instance_of(type)),
    )
    value: Optional[Any] = _attrs.field(default=None)

    def serialize(self) -> dict:
        # Inherit docstring
        return {"type": self.type, "value": self.value}

    @classmethod
    def deserialize(cls, obj):
        # Inherit docstring
        return cls(**obj)

    def validate(self, attr: Any):
        # Inherit docstring

        if self.type is not None:
            if not isinstance(attr, self.type):
                raise SchemaError(
                    f"attribute type mismatch {attr} is not of type {self.type}"
                )

        if self.value is not None:
            if self.value is not None and self.value != attr:
                raise SchemaError(f"name {attr} != {self.value}")


@_attrs.define(on_setattr=[_attrs.setters.convert, _attrs.setters.validate])
class AttrsSchema(BaseSchema):
    """
    Attributes schema

    Parameters
    ----------
    attrs : str or iterable of str
        Attributes definition

    require_all_keys : bool
        Whether to require to all coordinates included in ``attrs``.

    allow_extra_keys : bool
        Whether to allow coordinates not included in ``attrs`` dict.
    """

    attrs: Dict[Hashable, AttrSchema] = _attrs.field(converter=dict)
    require_all_keys: bool = _attrs.field(default=True)
    allow_extra_keys: bool = _attrs.field(default=True)

    def serialize(self) -> dict:
        # Inherit docstring
        obj = {
            "require_all_keys": self.require_all_keys,
            "allow_extra_keys": self.allow_extra_keys,
            "attrs": {k: v.serialize() for k, v in self.attrs.items()},
        }
        return obj

    @classmethod
    def deserialize(cls, obj: dict):
        # Inherit docstring
        if "attrs" in obj:
            attrs = obj["attrs"]
            kwargs = {k: v for k, v in obj.items() if k != "attrs"}
        else:
            attrs = obj
            kwargs = {}

        attrs = {k: AttrSchema.convert(v) for k, v in list(attrs.items())}
        return cls(attrs, **kwargs)

    def validate(self, attrs: Any) -> None:
        # Inherit docstring

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
