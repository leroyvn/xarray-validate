import json
from abc import abstractmethod
from typing import Any, Dict


class SchemaError(Exception):
    """Custom Schema Error"""


class BaseSchema:
    _json_schema: Dict[str, Any]

    @property
    @abstractmethod
    def json(self) -> Any:
        pass

    def to_json(self, **dumps_kws) -> str:
        """
        Generate a JSON string representation of this schema

        Parameters
        ----------
        dumps_kws : dict
            Parameters to pass to ``json.dumps``.

        Returns
        -------
        str
        """
        return json.dumps(self.json, **dumps_kws)

    @classmethod
    @abstractmethod
    def from_json(cls, obj):
        pass
