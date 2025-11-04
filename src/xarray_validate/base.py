from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List

import attrs


class ValidationMode(Enum):
    """
    Validation behaviour mode.
    """

    EAGER = "eager"  #: Raise SchemaError on first validation failure (default behavior)
    LAZY = "lazy"  #: Collect all validation errors and return them in ValidationResult


@attrs.define
class ValidationResult:
    """
    Result of schema validation with error location mapping.

    Parameters
    ----------
    errors : list of tuple[str, SchemaError]
        List of (path, error) pairs mapping errors to tree locations.
    """

    errors: list[tuple[str, SchemaError]] = attrs.field(factory=list)

    @property
    def has_errors(self):
        return bool(len(self.errors))

    def add_error(self, path: str, error: SchemaError) -> None:
        """Add an error at the specified path."""
        self.errors.append((path, error))

    def get_error_summary(self) -> str:
        """Get a formatted summary of all validation errors."""
        if not self.has_errors:
            return "Validation passed"

        lines = ["Validation failed with errors:"]
        for path, error in self.errors:
            lines.append(f"  {path}: {error}")
        return "\n".join(lines)


@attrs.define
class ValidationContext:
    """
    Context for tracking validation state during schema tree traversal.

    Parameters
    ----------
    path : list of str, optional
        Current validation path through the schema tree.

    mode : ValidationMode or str, default: :data:`ValidationMode.EAGER`
        Validation behavior mode (eager or lazy). Strings are converted to
        lowercase and passed to the :class:`ValidationMode` constructor.

    result : ValidationResult, optional
        Shared result object for collecting errors in lazy mode.
    """

    path: list[str] = attrs.field(factory=list, converter=list)
    mode: ValidationMode = attrs.field(
        default=ValidationMode.EAGER,
        converter=lambda x: ValidationMode(x.lower() if isinstance(x, str) else x),
    )
    result: ValidationResult = attrs.field(factory=ValidationResult)

    def push(self, component: str) -> ValidationContext:
        """
        Create a new context with an additional path component.

        Parameters
        ----------
        component : str
            Path component to add (e.g., 'dtype', 'coords.x', 'data_vars.temperature')

        Returns
        -------
        ValidationContext
            New context with extended path sharing the same mode and result.
        """
        return ValidationContext(
            path=self.path + [component], mode=self.mode, result=self.result
        )

    def get_path_string(self) -> str:
        """Get current path as dot-separated string."""
        return ".".join(self.path) if self.path else "<root>"

    def handle_error(self, error: SchemaError) -> None:
        """
        Handle validation error based on mode.

        * In EAGER mode: raise the error immediately
        * In LAZY mode: collect error in result object

        Parameters
        ----------
        error : SchemaError
            Validation error to handle.
        """
        if self.mode == ValidationMode.EAGER:
            raise error
        else:  # LAZY mode
            self.result.add_error(self.get_path_string(), error)

    def get_errors(self) -> List[tuple[str, SchemaError]]:
        """Get all collected errors with their paths."""
        return self.result.errors.copy()

    @property
    def has_errors(self) -> bool:
        """Check if any errors have been collected."""
        return self.result.has_errors


class SchemaError(Exception):
    """Custom schema error."""


class BaseSchema(ABC):
    @abstractmethod
    def serialize(self):
        """
        Serialize schema to basic Python types.
        """
        pass

    @classmethod
    @abstractmethod
    def deserialize(cls, obj):
        """
        Instantiate schema from basic Python types.
        """
        pass

    @classmethod
    def convert(cls, value: Any):
        """
        Attempt conversion of ``value`` to this schema type.
        """
        if isinstance(value, cls):
            return value
        return cls.deserialize(value)

    @abstractmethod
    def validate(self, value: Any, context: ValidationContext | None = None) -> None:
        """
        Validate object against this schema.

        Parameters
        ----------
        value
            Object to validate.

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
        pass
