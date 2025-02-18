import numpy as np
import pytest
import xarray as xr

from xarray_validate import DataArraySchema, DatasetSchema
from xarray_validate.base import SchemaError
from xarray_validate.components import AttrSchema, AttrsSchema


def test_dataset_empty_constructor():
    ds_schema = DatasetSchema()
    assert hasattr(ds_schema, "validate")
    assert ds_schema.serialize() == {
        "attrs": {},
        "data_vars": {},
    }  # TODO: Check for correctness


def test_dataset_example(ds):
    ds_schema = DatasetSchema(
        {
            "foo": DataArraySchema(name="foo", dtype=np.int32, dims=["x"]),
            "bar": DataArraySchema(name="bar", dtype=np.floating, dims=["x", "y"]),
        }
    )

    assert list(ds_schema.serialize()["data_vars"].keys()) == ["foo", "bar"]
    ds_schema.validate(ds)

    ds["foo"] = ds.foo.astype("float32")
    with pytest.raises(SchemaError, match="dtype"):
        ds_schema.validate(ds)

    ds = ds.drop_vars("foo")
    with pytest.raises(SchemaError, match="variable foo"):
        ds_schema.validate(ds)


def test_checks_ds(ds):
    def check_foo(ds):
        assert "foo" in ds

    ds_schema = DatasetSchema(checks=[check_foo])
    ds_schema.validate(ds)

    ds = ds.drop_vars("foo")
    with pytest.raises(AssertionError):
        ds_schema.validate(ds)

    ds_schema = DatasetSchema(checks=[])
    ds_schema.validate(ds)

    # TODO
    # with pytest.raises(ValueError):
    #     DatasetSchema(checks=[2])


def test_dataset_with_attrs_schema():
    name = "name"
    expected_value = "expected_value"
    actual_value = "actual_value"
    ds = xr.Dataset(attrs={name: actual_value})
    ds_schema = DatasetSchema(attrs={name: AttrSchema(value=expected_value)})

    ds_schema_2 = DatasetSchema(
        attrs=AttrsSchema({name: AttrSchema(value=expected_value)})
    )
    with pytest.raises(SchemaError):
        ds_schema.validate(ds)
    with pytest.raises(SchemaError):
        ds_schema_2.validate(ds)


def test_attrs_extra_key():
    name = "name"
    value = "value_2"
    name_2 = "name_2"
    value_2 = "value_2"
    ds = xr.Dataset(attrs={name: value})
    ds_schema = DatasetSchema(
        attrs=AttrsSchema(
            attrs={
                name: AttrSchema(
                    value=value,
                ),
                name_2: AttrSchema(value=value_2),
            },
            require_all_keys=True,
        )
    )

    with pytest.raises(SchemaError):
        ds_schema.validate(ds)


def test_attrs_missing_key():
    name = "name"
    value = "value_2"
    name_2 = "name_2"
    value_2 = "value_2"
    ds = xr.Dataset(attrs={name: value, name_2: value_2})
    ds_schema = DatasetSchema(
        attrs=AttrsSchema(attrs={name: AttrSchema(value=value)}, allow_extra_keys=False)
    )
    with pytest.raises(SchemaError):
        ds_schema.validate(ds)


def test_schema_from_dataset(ds):
    schema = DatasetSchema.from_dataset(ds)
    schema.validate(ds)

    expected = {
        "data_vars": {
            "foo": {
                "dtype": "<i4",
                "dims": ["x"],
                "shape": [4],
                "attrs": {
                    "require_all_keys": True,
                    "allow_extra_keys": True,
                    "attrs": {},
                },
            },
            "bar": {
                "dtype": "<f8",
                "dims": ["x", "y"],
                "shape": [4, 2],
                "attrs": {
                    "require_all_keys": True,
                    "allow_extra_keys": True,
                    "attrs": {},
                },
            },
        },
        "attrs": {"require_all_keys": True, "allow_extra_keys": True, "attrs": {}},
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
    }
    assert schema.serialize() == expected
