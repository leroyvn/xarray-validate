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
    assert da_schema.serialize() == {}
    da_schema.validate(da)


@pytest.mark.parametrize(
    "attribute_name, component_schema_cls, component_schema_args",
    [
        ("dtype", DTypeSchema, "i4"),
        ("dims", DimsSchema, ("x", None)),
        ("shape", ShapeSchema, (2, None)),
        ("name", NameSchema, "foo"),
        ("array_type", ArrayTypeSchema, np.ndarray),
        ("chunks", ChunksSchema, False),
    ],
)
def test_dataarray_component_constructors(
    attribute_name, component_schema_cls, component_schema_args
):
    da = xr.DataArray(np.zeros((2, 4), dtype="i4"), dims=("x", "y"), name="foo")
    component_schema = component_schema_cls(component_schema_args)
    data_array_schema = DataArraySchema(**{attribute_name: component_schema_args})
    assert (
        component_schema.serialize()
        == getattr(data_array_schema, attribute_name).serialize()
    )
    assert isinstance(getattr(data_array_schema, attribute_name), component_schema_cls)

    # json roundtrip
    rt_schema = DataArraySchema.deserialize(data_array_schema.serialize())
    assert isinstance(rt_schema, DataArraySchema)
    assert rt_schema.serialize() == data_array_schema.serialize()

    data_array_schema.validate(da)


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


def test_schema_from_dataarray(ds):
    da = ds["x"]

    schema = DataArraySchema.from_dataarray(da)
    schema.validate(da)
    
    expected = {
        "dtype": "<i8",
        "dims": ["x"],
        "shape": [4],
        "coords": {
            "require_all_keys": True,
            "allow_extra_keys": True,
            "coords": {
                "x": {
                    "dtype": "<i8",
                    "dims": ["x"],
                    "shape": [4],
                    "attrs": {
                        "require_all_keys": True,
                        "allow_extra_keys": True,
                        "attrs": {},
                    },
                }
            },
        },
        "name": "x",
        "attrs": {"require_all_keys": True, "allow_extra_keys": True, "attrs": {}},
    }
    assert schema.serialize() == expected
