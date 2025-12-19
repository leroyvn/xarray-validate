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
        "require_all_keys": True,
        "allow_extra_keys": True,
        "attrs": {},
        "data_vars": {},
    }


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
    with pytest.raises(SchemaError, match="data_vars has missing keys"):
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
    with pytest.raises(SchemaError, match="attrs has extra keys"):
        ds_schema.validate(ds)


def test_schema_from_dataset(ds):
    schema = DatasetSchema.from_dataset(ds)
    schema.validate(ds)

    expected = {
        "require_all_keys": True,
        "allow_extra_keys": True,
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


def test_optional_data_vars(ds):
    """Test that data variables can be made optional with require_all_keys=False."""
    ds_schema = DatasetSchema(
        data_vars={
            "foo": DataArraySchema(name="foo", dtype=np.int32, dims=["x"]),
            "bar": DataArraySchema(name="bar", dtype=np.floating, dims=["x", "y"]),
            "optional_var": DataArraySchema(dtype=np.float32),
        },
        require_all_keys=False,
    )

    # Original dataset should validate
    ds_schema.validate(ds)

    # Dataset missing "optional_var" should still validate
    ds_schema.validate(ds)

    # Dataset with only "foo" should validate
    ds_partial = ds.drop_vars("bar")
    ds_schema.validate(ds_partial)

    # Dataset with only "bar" should validate
    ds_partial = ds.drop_vars("foo")
    ds_schema.validate(ds_partial)

    # Empty dataset (no data vars from schema) should validate
    ds_empty = ds.drop_vars(["foo", "bar"])
    ds_schema.validate(ds_empty)


def test_data_vars_missing_keys_error():
    """Test that missing required data variables raise an error."""
    ds = xr.Dataset(
        {
            "foo": xr.DataArray([1, 2, 3], dims=["x"]),
        }
    )

    # Default behavior: require_all_keys=True
    ds_schema = DatasetSchema(
        data_vars={
            "foo": DataArraySchema(dtype=np.int64),
            "bar": DataArraySchema(dtype=np.float64),
        }
    )

    with pytest.raises(SchemaError, match="data_vars has missing keys"):
        ds_schema.validate(ds)


def test_data_vars_extra_keys_error():
    """Test that extra data variables raise an error when allow_extra_keys=False."""
    ds = xr.Dataset(
        {
            "foo": xr.DataArray([1, 2, 3], dims=["x"]),
            "bar": xr.DataArray([1.0, 2.0, 3.0], dims=["x"]),
            "extra": xr.DataArray([4, 5, 6], dims=["x"]),
        }
    )

    ds_schema = DatasetSchema(
        data_vars={
            "foo": DataArraySchema(dtype=np.int64),
            "bar": DataArraySchema(dtype=np.float64),
        },
        allow_extra_keys=False,
    )

    with pytest.raises(SchemaError, match="data_vars has extra keys"):
        ds_schema.validate(ds)


def test_glob_pattern_data_vars():
    """Test glob pattern matching for data variables."""
    ds = xr.Dataset(
        {
            "x_0": xr.DataArray([1.0, 2.0, 3.0], dims=["time"]),
            "x_1": xr.DataArray([4.0, 5.0, 6.0], dims=["time"]),
            "x_2": xr.DataArray([7.0, 8.0, 9.0], dims=["time"]),
            "temp": xr.DataArray([10.0, 11.0, 12.0], dims=["time"]),
        }
    )

    # Pattern key 'x_*' should match all variables starting with 'x_'
    ds_schema = DatasetSchema(
        data_vars={
            "x_*": DataArraySchema(dtype=np.float64, dims=["time"]),
            "temp": DataArraySchema(dtype=np.float64, dims=["time"]),
        }
    )

    # Should validate successfully
    ds_schema.validate(ds)


def test_regex_pattern_data_vars():
    """Test regex pattern matching for data variables."""
    ds = xr.Dataset(
        {
            "x_0": xr.DataArray([1.0, 2.0, 3.0], dims=["time"]),
            "x_1": xr.DataArray([4.0, 5.0, 6.0], dims=["time"]),
            "x_foo": xr.DataArray([7.0, 8.0, 9.0], dims=["time"]),
            "temp": xr.DataArray([10.0, 11.0, 12.0], dims=["time"]),
        }
    )

    # Regex pattern '{x_\d+}' should match only x_0, x_1 but not x_foo
    ds_schema = DatasetSchema(
        data_vars={
            r"{x_\d+}": DataArraySchema(dtype=np.float64, dims=["time"]),
            "temp": DataArraySchema(dtype=np.float64, dims=["time"]),
            "x_foo": DataArraySchema(dtype=np.float64, dims=["time"]),
        }
    )

    # Should validate successfully
    ds_schema.validate(ds)


def test_glob_pattern_multiple_patterns():
    """Test multiple glob patterns in data_vars."""
    ds = xr.Dataset(
        {
            "x_0": xr.DataArray([1.0, 2.0], dims=["time"]),
            "x_1": xr.DataArray([3.0, 4.0], dims=["time"]),
            "y_0": xr.DataArray([5.0, 6.0], dims=["time"]),
            "y_1": xr.DataArray([7.0, 8.0], dims=["time"]),
            "z": xr.DataArray([9.0, 10.0], dims=["time"]),
        }
    )

    ds_schema = DatasetSchema(
        data_vars={
            "x_*": DataArraySchema(dtype=np.float64, dims=["time"]),
            "y_*": DataArraySchema(dtype=np.float64, dims=["time"]),
            "z": DataArraySchema(dtype=np.float64, dims=["time"]),
        }
    )

    ds_schema.validate(ds)


def test_pattern_with_allow_extra_keys_false():
    """Test patterns with allow_extra_keys=False."""
    ds = xr.Dataset(
        {
            "x_0": xr.DataArray([1.0, 2.0], dims=["time"]),
            "x_1": xr.DataArray([3.0, 4.0], dims=["time"]),
            "temp": xr.DataArray([5.0, 6.0], dims=["time"]),
        }
    )

    # 'x_*' matches x_0 and x_1, 'temp' is exact match
    ds_schema = DatasetSchema(
        data_vars={
            "x_*": DataArraySchema(dtype=np.float64),
            "temp": DataArraySchema(dtype=np.float64),
        },
        allow_extra_keys=False,
    )

    # Should validate successfully
    ds_schema.validate(ds)

    # Add an extra variable that doesn't match any pattern
    ds["extra"] = xr.DataArray([7.0, 8.0], dims=["time"])

    # Should raise error for extra key
    with pytest.raises(SchemaError, match="data_vars has extra keys"):
        ds_schema.validate(ds)


def test_pattern_validation_error():
    """Test that validation errors are raised for variables matching patterns."""
    ds = xr.Dataset(
        {
            "x_0": xr.DataArray([1.0, 2.0], dims=["time"]),
            "x_1": xr.DataArray([3, 4], dims=["time"]),  # Wrong dtype
        }
    )

    ds_schema = DatasetSchema(
        data_vars={
            "x_*": DataArraySchema(dtype=np.float64),
        }
    )

    # Should raise error because x_1 has wrong dtype
    with pytest.raises(SchemaError, match="dtype"):
        ds_schema.validate(ds)


def test_pattern_require_all_keys():
    """Test that require_all_keys only applies to exact keys, not patterns."""
    ds = xr.Dataset(
        {
            "temp": xr.DataArray([1.0, 2.0], dims=["time"]),
            # No x_* variables present
        }
    )

    # Pattern keys should be optional even with require_all_keys=True
    ds_schema = DatasetSchema(
        data_vars={
            "x_*": DataArraySchema(dtype=np.float64),
            "temp": DataArraySchema(dtype=np.float64),
        },
        require_all_keys=True,
    )

    # Should validate successfully (pattern keys are optional)
    ds_schema.validate(ds)

    # But exact keys should still be required
    ds_no_temp = xr.Dataset({})
    with pytest.raises(SchemaError, match="data_vars has missing keys"):
        ds_schema.validate(ds_no_temp)


def test_pattern_exact_key_precedence():
    """Test that exact keys take precedence over pattern keys."""
    ds = xr.Dataset(
        {
            "x_special": xr.DataArray([1, 2], dims=["time"]),
            "x_0": xr.DataArray([3.0, 4.0], dims=["time"]),
        }
    )

    # 'x_special' has both an exact match and matches the pattern 'x_*'
    # The exact key schema should be used
    ds_schema = DatasetSchema(
        data_vars={
            "x_special": DataArraySchema(dtype=np.int64),  # Exact match: int64
            "x_*": DataArraySchema(dtype=np.float64),  # Pattern: float64
        }
    )

    # Should validate successfully: x_special uses exact schema (int64)
    # and x_0 uses pattern schema (float64)
    ds_schema.validate(ds)


def test_regex_vs_glob_patterns():
    """Test that regex patterns are more restrictive than glob patterns."""
    ds = xr.Dataset(
        {
            "var_0": xr.DataArray([1.0, 2.0], dims=["time"]),
            "var_1": xr.DataArray([3.0, 4.0], dims=["time"]),
            "var_foo": xr.DataArray([5.0, 6.0], dims=["time"]),
        }
    )

    # Glob pattern 'var_*' matches all
    glob_schema = DatasetSchema(
        data_vars={
            "var_*": DataArraySchema(dtype=np.float64),
        }
    )
    glob_schema.validate(ds)

    # Regex pattern '{var_\d+}' matches only var_0 and var_1
    # This should fail because var_foo doesn't match the pattern
    regex_schema = DatasetSchema(
        data_vars={
            r"{var_\d+}": DataArraySchema(dtype=np.float64),
        },
        allow_extra_keys=False,
    )
    with pytest.raises(SchemaError, match="data_vars has extra keys"):
        regex_schema.validate(ds)
