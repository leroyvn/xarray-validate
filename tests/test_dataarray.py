import jsonschema
import numpy as np
import pytest
import xarray as xr
from attrs.exceptions import NotCallableError

from xarray_validate import (
    ArrayTypeSchema,
    ChunksSchema,
    DataArraySchema,
    DimsSchema,
    DTypeSchema,
    NameSchema,
    ShapeSchema,
)


def test_dataarray_empty_constructor():
    da = xr.DataArray(np.ones(4, dtype="i4"))
    da_schema = DataArraySchema()
    assert hasattr(da_schema, "validate")
    jsonschema.validate(da_schema.json, da_schema._json_schema)
    assert da_schema.json == {}
    da_schema.validate(da)


@pytest.mark.parametrize(
    "kind, component, schema_args",
    [
        ("dtype", DTypeSchema, "i4"),
        ("dims", DimsSchema, ("x", None)),
        ("shape", ShapeSchema, (2, None)),
        ("name", NameSchema, "foo"),
        ("array_type", ArrayTypeSchema, np.ndarray),
        ("chunks", ChunksSchema, False),
    ],
)
def test_dataarray_component_constructors(kind, component, schema_args):
    da = xr.DataArray(np.zeros((2, 4), dtype="i4"), dims=("x", "y"), name="foo")
    comp_schema = component(schema_args)
    schema = DataArraySchema(**{kind: schema_args})
    assert comp_schema.json == getattr(schema, kind).json
    jsonschema.validate(schema.json, schema._json_schema)
    assert isinstance(getattr(schema, kind), component)

    # json roundtrip
    rt_schema = DataArraySchema.from_json(schema.json)
    assert isinstance(rt_schema, DataArraySchema)
    assert rt_schema.json == schema.json

    schema.validate(da)


def test_dataarray_schema_validate_raises_for_invalid_input_type():
    ds = xr.Dataset()
    schema = DataArraySchema()
    with pytest.raises(ValueError, match="Input must be a xarray.DataArray"):
        schema.validate(ds)


def test_checks_da(ds):
    da = ds["foo"]

    def check_foo(da):
        assert da.name == "foo"

    def check_bar(da):
        assert da.name == "bar"

    schema = DataArraySchema(checks=[check_foo])
    schema.validate(da)

    schema = DataArraySchema(checks=[check_bar])
    with pytest.raises(AssertionError):
        schema.validate(da)

    schema = DataArraySchema(checks=[])
    schema.validate(da)

    with pytest.raises(NotCallableError):
        DataArraySchema(checks=[2])
