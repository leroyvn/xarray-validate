from __future__ import annotations

import fnmatch
import re
from typing import Callable, Dict, Iterable, Literal, Optional, Union

import attrs as _attrs
import xarray as xr

from .base import (
    BaseSchema,
    SchemaError,
    ValidationContext,
    ValidationMode,
)
from .components import AttrsSchema
from .dataarray import CoordsSchema, DataArraySchema


def _is_regex_pattern(key: str) -> bool:
    """Check if a key is a regex pattern (enclosed in curly braces)."""
    return key.startswith("{") and key.endswith("}")


def _is_glob_pattern(key: str) -> bool:
    """Check if a key is a glob pattern (contains * or ?)."""
    return "*" in key or "?" in key


def _is_pattern_key(key: str) -> bool:
    """Check if a key is any kind of pattern (glob or regex)."""
    return _is_glob_pattern(key) or _is_regex_pattern(key)


def _pattern_to_regex(pattern: str) -> re.Pattern:
    """
    Convert a pattern key to a compiled regex.

    Supports two pattern types:
    - Glob patterns: 'x_*' matches x_0, x_1, x_foo, etc.
    - Regex patterns: '{x_\\d+}' matches x_0, x_1, but not x_foo

    Parameters
    ----------
    pattern : str
        The pattern string (glob or regex in curly braces)

    Returns
    -------
    re.Pattern
        Compiled regex pattern
    """
    if _is_regex_pattern(pattern):
        # Remove curly braces and compile as regex
        regex_str = pattern[1:-1]
        return re.compile(regex_str)
    elif _is_glob_pattern(pattern):
        # Convert glob to regex
        regex_str = fnmatch.translate(pattern)
        return re.compile(regex_str)
    else:
        # Exact match
        return re.compile(re.escape(pattern) + "$")


@_attrs.define(on_setattr=[_attrs.setters.convert, _attrs.setters.validate])
class DatasetSchema(BaseSchema):
    r"""
    A lightweight xarray.Dataset validator.

    Parameters
    ----------
    data_vars : dict, optional
        Per-variable :class:`.DataArraySchema`\ s. Keys can be either exact
        variable names or patterns:

        - Exact match: ``'temperature'`` matches only 'temperature'
        - Glob pattern: ``'x_*'`` matches x_0, x_1, x_foo, etc.
        - Regex pattern: ``'{x_\\d+}'`` matches x_0, x_1, but not x_foo

    require_all_keys : bool, default: True
        Whether to require all data variables included in ``data_vars``.
        Only applies to exact keys, not pattern keys.

    allow_extra_keys : bool, default: True
        Whether to allow data variables not included in ``data_vars`` dict.
        Variables matching pattern keys are not considered "extra".

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

    require_all_keys: bool = _attrs.field(default=True)
    allow_extra_keys: bool = _attrs.field(default=True)

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
            "require_all_keys": self.require_all_keys,
            "allow_extra_keys": self.allow_extra_keys,
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

        if "require_all_keys" in obj:
            kwargs["require_all_keys"] = obj["require_all_keys"]
        if "allow_extra_keys" in obj:
            kwargs["allow_extra_keys"] = obj["allow_extra_keys"]
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

    def validate(
        self,
        ds: xr.Dataset,
        context: ValidationContext | None = None,
        mode: Literal["eager", "lazy"] | None = None,
    ) -> None:
        """
        Validate an xarray.DataArray against this schema.

        Parameters
        ----------
        ds : Dataset
            Dataset to validate.

        context : ValidationContext, optional
            Validation context for tracking tree traversal state.

        mode : {"eager", "lazy"}, optional
            Validation mode. If unset, the global default mode (eager) is used.

        Returns
        -------
        ValidationResult or None
            In eager mode, this method returns ``None``. In lazy mode, it
            returns a :class:`ValidationResult` object.
        """

        if mode is None:
            mode = "eager"

        if context is None:
            context = ValidationContext(mode=mode)

        if self.data_vars is not None:
            # Separate exact keys from pattern keys
            exact_keys = {k: v for k, v in self.data_vars.items() if not _is_pattern_key(k)}
            pattern_keys = {k: v for k, v in self.data_vars.items() if _is_pattern_key(k)}

            # Compile pattern regexes
            compiled_patterns = {k: _pattern_to_regex(k) for k in pattern_keys}

            if self.require_all_keys:
                # Only check exact keys for require_all_keys
                missing_keys = set(exact_keys.keys()) - set(ds.data_vars.keys())
                if missing_keys:
                    error = SchemaError(f"data_vars has missing keys: {missing_keys}")
                    if context:
                        context.handle_error(error)
                    else:
                        raise error

            if not self.allow_extra_keys:
                # Check that all dataset variables match either exact or pattern keys
                matched_vars = set()
                for var_name in ds.data_vars.keys():
                    # Check exact match
                    if var_name in exact_keys:
                        matched_vars.add(var_name)
                        continue
                    # Check pattern match
                    for pattern, regex in compiled_patterns.items():
                        if regex.fullmatch(var_name):
                            matched_vars.add(var_name)
                            break

                extra_keys = set(ds.data_vars.keys()) - matched_vars
                if extra_keys:
                    error = SchemaError(f"data_vars has extra keys: {extra_keys}")
                    if context:
                        context.handle_error(error)
                    else:
                        raise error

            # Validate variables matching exact keys
            for key, da_schema in exact_keys.items():
                if da_schema is not None and key in ds.data_vars:
                    data_var_context = context.push(f"data_vars.{key}")
                    da_schema.validate(ds.data_vars[key], data_var_context)

            # Validate variables matching pattern keys
            for pattern_key, da_schema in pattern_keys.items():
                if da_schema is None:
                    continue
                regex = compiled_patterns[pattern_key]
                for var_name in ds.data_vars.keys():
                    if regex.fullmatch(var_name) and var_name not in exact_keys:
                        data_var_context = context.push(f"data_vars.{var_name}")
                        da_schema.validate(ds.data_vars[var_name], data_var_context)

        if self.coords is not None:  # pragma: no cover
            coords_context = context.push("coords")
            self.coords.validate(ds.coords, coords_context)

        if self.attrs:
            attrs_context = context.push("attrs")
            self.attrs.validate(ds.attrs, attrs_context)

        if self.checks:
            for check in self.checks:
                check(ds)

        return None if context.mode is ValidationMode.EAGER else context.result
