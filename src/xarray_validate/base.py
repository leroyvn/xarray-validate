from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


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
    def validate(self, value: Any) -> None:
        """
        Validate object against this schema.

        Raises
        ------
        SchemaError
            If validation fails.
        """
