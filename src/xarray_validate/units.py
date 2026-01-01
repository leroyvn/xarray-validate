from __future__ import annotations

from typing import TYPE_CHECKING

from .base import SchemaError, ValidationContext, raise_or_handle

if TYPE_CHECKING:
    import pint

_REGISTRY: pint.UnitRegistry | None = None


try:
    import pint
except ImportError as e:
    raise ImportError(
        "Unit validation requires the pint library. Install with pip install pint"
    ) from e


def set_registry(ureg: pint.UnitRegistry | None = None) -> None:
    global _REGISTRY
    _REGISTRY = ureg if ureg is not None else pint.get_application_registry()


def get_registry() -> pint.UnitRegistry:
    """
    Get the default unit registry.

    If not set by the user using :func:`.set_registry`,"""
    if _REGISTRY is None:
        set_registry()
    return _REGISTRY


def parse(
    unit_string: str,
    ureg: pint.UnitRegistry | None = None,
    context: ValidationContext | None = None,
    error_prefix: str = "Invalid units",
):
    """
    Parse a unit string with pint, handling errors appropriately.

    Parameters
    ----------
    unit_string : str
        The unit string to parse.

    ureg : pint.UnitRegistry, optional
        The pint unit registry to use for parsing. If not passed, the default
        registry is used.

    context : ValidationContext, optional
        Validation context for error handling.

    error_prefix : str, default: "Invalid units"
        Prefix for error messages.

    Returns
    -------
    pint.Unit or None
        The parsed unit, or None if parsing failed.
    """
    if ureg is None:
        ureg = get_registry()

    try:
        return ureg.Unit(unit_string)
    except (pint.UndefinedUnitError, pint.errors.DefinitionSyntaxError) as e:
        error = SchemaError(f"{error_prefix} '{unit_string}': {e}")
        raise_or_handle(error, context, from_exc=e)
        return None
