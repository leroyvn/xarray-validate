from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict


class SchemaError(Exception):
    """Custom Schema Error"""


class BaseSchema(ABC):
    _json_schema: ClassVar[Dict[str, Any]]

    @property
    @abstractmethod
    def json(self):
        """
        Return a representation of the schema as a JSON-serializable object.
        """
        pass

    def to_json(self, **dumps_kws) -> str:
        """
        Generate a JSON string representation of this schema.

        Parameters
        ----------
        dumps_kws : dict
            Keyword arguments forwarded to pass to :func:`json.dumps`.

        Returns
        -------
        str
        """
        return json.dumps(self.json, **dumps_kws)

    @classmethod
    @abstractmethod
    def from_json(cls, obj):
        """
        Instantiate from a mapping containing the JSON representation of the
        object.
        """
        pass
