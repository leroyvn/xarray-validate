from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List

import attrs


@attrs.define
class ValidationContext:
    """
    Context for tracking validation state during schema tree traversal.

    This class maintains state about the current validation path and can be
    extended to support lazy error reporting in the future.

    Parameters
    ----------
    path : list of str, optional
        Current validation path through the schema tree.
    """

    path: list[str] = attrs.field(factory=list, converter=list)
    _errors: list = attrs.field(init=False, factory=list)

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
            New context with extended path.
        """
        return ValidationContext(self.path + [component])

    def get_path_string(self) -> str:
        """Get current path as dot-separated string."""
        return ".".join(self.path) if self.path else "<root>"

    def add_error(self, error: SchemaError) -> None:
        """
        Add validation error to context (for future lazy reporting).

        Parameters
        ----------
        error : SchemaError
            Validation error to record
        """
        self._errors.append((self.get_path_string(), error))

    def get_errors(self) -> List[tuple[str, SchemaError]]:
        """Get all collected errors with their paths."""
        return self._errors.copy()

    def has_errors(self) -> bool:
        """Check if any errors have been collected."""
        return len(self._errors) > 0


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
        value : Any
            Object to validate

        context : ValidationContext, optional
            Validation context for tracking tree traversal state

        Raises
        ------
        SchemaError
            If validation fails.
        """
