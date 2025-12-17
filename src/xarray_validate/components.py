from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Dict, Hashable, Optional, Tuple, Union

import attrs as _attrs
import numpy as np
from numpy.typing import DTypeLike

from . import converters
from .base import BaseSchema, SchemaError, ValidationContext
from .types import ChunksT, DimsT, ShapeT


def _dtype_converter(value: DTypeLike):
    if isinstance(value, (tuple, list)):
        return tuple(_dtype_converter(x) for x in value)

    if value in {np.integer, np.floating}:
        return value

    if value == "integer":
        return np.integer

    if value == "floating":
        return np.floating

    return np.dtype(value)


@_attrs.define(on_setattr=[_attrs.setters.convert, _attrs.setters.validate])
class DTypeSchema(BaseSchema):
    """
    Data type schema.

    This schema type validates NumPy's data type objects.

    Parameters
    ----------
    dtype : DTypeLike
        DataArray dtype. Generic dtypes ``np.integer`` and ``np.floating`` are
        supported.

    Examples
    --------
    Basic instantiation and validation work like this:

    >>> schema = DTypeSchema("float64")
    >>> schema.validate(np.dtype("float64"))

    Validation uses :func:`numpy.issubdtype` and is therefore strict:

    >>> schema.validate(np.dtype("float32"))
    Traceback (most recent call last):
    ...
    SchemaError: dtype mismatch: got dtype('float32'), expected dtype('float64')

    Generics are supported. For instance:

    >>> DTypeSchema("floating").validate(np.ones(5, dtype="float16").dtype)
    >>> DTypeSchema("integer").validate(np.ones(5, dtype="int32").dtype)

    For flexibility, multiple dtypes can be passed. Validation will pass if
    at least one validates:

    >>> schema = DTypeSchema(["float64", "float32"])
    >>> schema.validate(np.dtype("float64"))
    >>> schema.validate(np.dtype("float32"))
    >>> schema.validate(np.dtype("float16"))
    Traceback (most recent call last):
    ...
    SchemaError: dtype mismatch: got dtype('float16'), expected one of
                 (dtype('float32'), dtype('float64'))
    """

    dtype: np.dtype | tuple[np.dtype, ...] = _attrs.field(converter=_dtype_converter)

    def serialize(self):
        # Inherit docstring
        return self.dtype.str

    @classmethod
    def deserialize(cls, obj):
        """
        Instantiate schema from a dtype-like object.
        """

        return cls(obj)

    def validate(
        self, dtype: DTypeLike, context: ValidationContext | None = None
    ) -> None:
        """
        Validate dtype against this schema.

        Parameters
        ----------
        dtype : DTypeLike
            Dtype to validate.

        context : ValidationContext, optional
            Validation context for tracking tree traversal state.

        Returns
        -------
        None

        Raises
        ------
        SchemaError
            If validation fails.
        """

        self_dtypes = self.dtype

        if not isinstance(self_dtypes, tuple):
            self_dtypes = (self_dtypes,)

        for self_dtype in self_dtypes:
            if np.issubdtype(dtype, self_dtype):
                passed = True
                break
        else:
            passed = False

        if not passed:
            msg = f"dtype mismatch: got {repr(dtype)}, expected " + (
                f"{repr(self_dtypes[0])}"
                if len(self_dtypes) == 1
                else f"one of {repr(self_dtypes)}"
            )
            error = SchemaError(msg)

            if context:
                context.handle_error(error)
            else:
                raise error


@_attrs.define(on_setattr=[_attrs.setters.convert, _attrs.setters.validate])
class DimsSchema(BaseSchema):
    """
    Dimensions schema.

    Parameters
    ----------
    dims : sequence of (str or None)
        DataArray dimensions. ``None`` may be used as a wildcard.

    ordered : bool, optional
        If ``False``, allow different dimension ordering.
    """

    dims: DimsT = _attrs.field(
        converter=lambda x: tuple(x) if not isinstance(x, str) else x,
        validator=_attrs.validators.deep_iterable(
            member_validator=_attrs.validators.optional(
                _attrs.validators.instance_of(str)
            )
        ),
    )

    ordered: bool = _attrs.field(default=True)

    def serialize(self) -> list | dict:
        # Inherit docstring
        dims = list(self.dims)
        if self.ordered:
            return dims
        else:
            return {"dims": dims, "ordered": bool(self.ordered)}

    @classmethod
    def deserialize(cls, obj: DimsT | dict) -> DimsSchema:
        """
        Instantiate schema from basic Python types.

        Two input types are supported:

        * a sequence of strings;
        * a dictionary that contains a ``dims`` entry (with a sequence of
          strings as value) and an optional ``ordered`` entry (with a boolean as
          value).
        """

        if isinstance(obj, Sequence):
            dims = obj
            kwargs = {}
        else:
            dims = obj["dims"]
            kwargs = {k: v for k, v in obj.items() if k != "dims"}

        return cls(dims, **kwargs)

    def validate(self, dims: DimsT, context: ValidationContext | None = None) -> None:
        """
        Validate dimensions against this schema.

        Parameters
        ----------
        dims : sequence of str
            Dimensions to validate.

        context : ValidationContext, optional
            Validation context for tracking tree traversal state.

        Returns
        -------
        None

        Raises
        ------
        SchemaError
            If validation fails.
        """

        if len(self.dims) != len(dims):
            error = SchemaError(
                f"dimension number mismatch: got {len(dims)}, expected {len(self.dims)}"
            )
            if context:
                context.handle_error(error)
            else:
                raise error

        if self.ordered:
            for i, (actual, expected) in enumerate(zip(dims, self.dims)):
                if expected is not None and actual != expected:
                    error = SchemaError(
                        f"dimension mismatch in axis {i}: got {actual}, "
                        f"expected {expected}"
                    )
                    if context:
                        context.handle_error(error)
                    else:
                        raise error
        else:
            for i, expected in enumerate(self.dims):
                if expected is not None and expected not in dims:
                    error = SchemaError(
                        f"dimension mismatch: expected {expected} is missing "
                        f"from actual dimension list {dims}"
                    )
                    if context:
                        context.handle_error(error)
                    else:
                        raise error


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
        """
        Instantiate schema from a sequence of integers (use None as a wildcard).
        """

        return cls(obj)

    def validate(self, shape: tuple, context: ValidationContext | None = None) -> None:
        """
        Validate shape against this schema.

        Parameters
        ----------
        shape : tuple of int
            Shape to validate.

        context : ValidationContext, optional
            Validation context for tracking tree traversal state.

        Returns
        -------
        None

        Raises
        ------
        SchemaError
            If validation fails.
        """

        if len(self.shape) != len(shape):
            error = SchemaError(
                "dimension count mismatch: "
                f"got {len(shape)}, expected {len(self.shape)}"
            )
            if context:
                context.handle_error(error)
            else:
                raise error

        for i, (actual, expected) in enumerate(zip(shape, self.shape)):
            if expected is not None and actual != expected:
                error = SchemaError(
                    f"shape mismatch in axis {i}: got {actual}, expected {expected}"
                )
                if context:
                    context.handle_error(error)
                else:
                    raise error


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
        """
        Instantiate schema from a string.
        """

        return cls(obj)

    def validate(self, name: str, context: ValidationContext | None = None) -> None:
        """
        Validate name against this schema.

        Parameters
        ----------
        name : str
            Name to validate.

        context : ValidationContext, optional
            Validation context for tracking tree traversal state.

        Returns
        -------
        None

        Raises
        ------
        SchemaError
            If validation fails.
        """

        # TODO: support regular expressions
        # - http://json-schema.org/understanding-json-schema/reference/regular_expressions.html
        # - https://docs.python.org/3.9/library/re.html
        if self.name != name:
            error = SchemaError(f"name mismatch: got {name}, expected {self.name}")
            if context:
                context.handle_error(error)
            else:
                raise error


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
        """
        Instantiate schema from a dictionary.
        """

        return cls(obj)

    def validate(
        self,
        chunks: Optional[Tuple[Tuple[int, ...], ...]],
        dims: Tuple,
        shape: Tuple[int, ...],
        context: ValidationContext | None = None,
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

        context : ValidationContext, optional
            Validation context for tracking tree traversal state.

        Returns
        -------
        None

        Raises
        ------
        SchemaError
            If validation fails.
        """

        if isinstance(self.chunks, bool):
            if self.chunks and not chunks:
                error = SchemaError("expected array to be chunked but it is not")
                if context:
                    context.handle_error(error)
                else:
                    raise error
            elif not self.chunks and chunks:
                error = SchemaError("expected unchunked array but it is chunked")
                if context:
                    context.handle_error(error)
                else:
                    raise error
        elif isinstance(self.chunks, dict):
            if chunks is None:
                error = SchemaError("expected array to be chunked but it is not")
                if context:
                    context.handle_error(error)
                else:
                    raise error
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
                        error = SchemaError(
                            f"chunk mismatch for {key}: got {ac}, expected {ec}"
                        )
                        if context:
                            context.handle_error(error)
                        else:
                            raise error

                else:  # assumes ec is an iterable
                    ac = dim_chunks[key]
                    if ec is not None and tuple(ac) != tuple(ec):
                        error = SchemaError(
                            f"chunk mismatch for {key}: got {ac}, expected {ec}"
                        )
                        if context:
                            context.handle_error(error)
                        else:
                            raise error
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
        """
        Instantiate schema from a string.
        """
        return cls(obj)

    def validate(self, array: Any, context: ValidationContext | None = None) -> None:
        """
        Validate array type against this schema.

        Parameters
        ----------
        array : Any
            Array to validate.

        context : ValidationContext, optional
            Validation context for tracking tree traversal state.


        Returns
        -------
        None

        Raises
        ------
        SchemaError
            If validation fails.
        """

        if not isinstance(array, self.array_type):
            error = SchemaError(
                f"array type mismatch: got {type(array)}, expected {self.array_type}"
            )
            if context:
                context.handle_error(error)
            else:
                raise error


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
        """
        Instantiate schema from a dictionary.
        """

        return cls(**obj)

    def validate(self, attr: Any, context: ValidationContext | None = None):
        """
        Validate attribute against this schema.

        Parameters
        ----------
        attr : Any
            Attribute value to validate.

        context : ValidationContext, optional
            Validation context for tracking tree traversal state.

        Returns
        -------
        None

        Raises
        ------
        SchemaError
            If validation fails.
        """

        if self.type is not None:
            if not isinstance(attr, self.type):
                error = SchemaError(
                    f"attribute type mismatch {attr} is not of type {self.type}"
                )
                if context:
                    context.handle_error(error)
                else:
                    raise error

        if self.value is not None:
            if self.value is not None and self.value != attr:
                error = SchemaError(f"name {attr} != {self.value}")
                if context:
                    context.handle_error(error)
                else:
                    raise error


@_attrs.define(on_setattr=[_attrs.setters.convert, _attrs.setters.validate])
class AttrsSchema(BaseSchema):
    """
    Attribute mapping schema.

    Parameters
    ----------
    attrs : dict
        Attribute definitions.

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
        """
        Instantiate schema from a dictionary.
        """

        if "attrs" in obj:
            attrs = obj["attrs"]
            kwargs = {k: v for k, v in obj.items() if k != "attrs"}
        else:
            attrs = obj
            kwargs = {}

        # None attribute definitions are allowed and will be converted to
        # AttrSchema()
        attrs = {
            k: (AttrSchema.convert(v) if v is not None else AttrSchema())
            for k, v in list(attrs.items())
        }
        return cls(attrs, **kwargs)

    def validate(self, attrs: Any, context: ValidationContext | None = None) -> None:
        """
        Validate attributes dictionary against this schema.

        Parameters
        ----------
        attrs : dict
            Attributes dictionary to validate.

        context : ValidationContext, optional
            Validation context for tracking tree traversal state.

        Returns
        -------
        None

        Raises
        ------
        SchemaError
            If validation fails.
        """

        if self.require_all_keys:
            missing_keys = set(self.attrs) - set(attrs)
            if missing_keys:
                error = SchemaError(f"attrs has missing keys: {missing_keys}")
                if context:
                    context.handle_error(error)
                else:
                    raise error

        if not self.allow_extra_keys:
            extra_keys = set(attrs) - set(self.attrs)
            if extra_keys:
                error = SchemaError(f"attrs has extra keys: {extra_keys}")
                if context:
                    context.handle_error(error)
                else:
                    raise error

        for key, attr_schema in self.attrs.items():
            if key not in attrs:
                error = SchemaError(f"key {key} not in attrs")
                if context:
                    context.handle_error(error)
                else:
                    raise error
            else:
                child_context = context.push(f"attrs.{key}") if context else None
                attr_schema.validate(attrs[key], child_context)
