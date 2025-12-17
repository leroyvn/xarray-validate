"""Testing helpers for xarray-validate."""

from __future__ import annotations

from typing import Type


def assert_construct(component: Type, schema_args):
    try:
        schema = component(schema_args)
    except TypeError:
        print(f"init of {component} from {schema_args} failed")
        raise

    return schema


def assert_json(schema, json):
    cls = type(schema)

    assert schema.serialize() == json, f"JSON export of {cls} failed"
    assert cls.deserialize(schema.serialize()).serialize() == json, (
        f"JSON roundtrip of {cls} failed"
    )
