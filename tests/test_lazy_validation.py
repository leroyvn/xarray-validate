"""Tests for lazy validation mode and error collection."""

import numpy as np
import pytest
import xarray as xr

from xarray_validate import (
    DataArraySchema,
    DatasetSchema,
    DimsSchema,
    DTypeSchema,
    NameSchema,
    SchemaError,
    ValidationContext,
    ValidationMode,
    ValidationResult,
)


class TestValidationMode:
    """Test ValidationMode enum."""

    def test_validation_mode_values(self):
        """Test that ValidationMode has expected values."""
        assert ValidationMode.EAGER.value == "eager"
        assert ValidationMode.LAZY.value == "lazy"


class TestValidationResult:
    """Test ValidationResult class."""

    def test_validation_result_init(self):
        result = ValidationResult()
        assert result.errors == []
        assert result.has_errors is False

    def test_add_errors(self):
        result = ValidationResult()
        result.add_error("path.1", SchemaError("Error 1"))
        result.add_error("path.2", SchemaError("Error 2"))
        assert result.has_errors is True
        assert len(result.errors) == 2

    def test_get_error_summary_valid(self):
        summary = ValidationResult().get_error_summary()
        assert summary == "Validation passed"

    def test_get_error_summary_invalid(self):
        result = ValidationResult([("test.path", SchemaError("Test error"))])
        summary = result.get_error_summary()
        assert "Validation failed with errors:" in summary
        assert "test.path: Test error" in summary


class TestValidationContext:
    """Test ValidationContext class."""

    def test_init_default(self):
        ctx = ValidationContext()
        assert ctx.path == []
        assert ctx.mode == ValidationMode.EAGER
        assert ctx.result.has_errors is False

    def test_init_with_path(self):
        ctx = ValidationContext(path=["a", "b"])
        assert ctx.path == ["a", "b"]

    def test_init_lazy_mode(self):
        ctx = ValidationContext(mode=ValidationMode.LAZY)
        assert ctx.mode == ValidationMode.LAZY
        ctx = ValidationContext(mode="Lazy")
        assert ctx.mode == ValidationMode.LAZY

    def test_push(self):
        ctx = ValidationContext(path=["a"])
        ctx2 = ctx.push("b")

        # Original context unchanged
        assert ctx.path == ["a"]
        # New context has extended path
        assert ctx2.path == ["a", "b"]
        # Shares mode and result
        assert ctx2.mode == ctx.mode
        assert ctx2.result is ctx.result

    def test_get_path_string_empty(self):
        ctx = ValidationContext()
        assert ctx.get_path_string() == "<root>"

    def test_get_path_string_with_path(self):
        ctx = ValidationContext(path=["data_vars", "temperature", "dtype"])
        assert ctx.get_path_string() == "data_vars.temperature.dtype"

    def test_handle_error_eager_mode(self):
        """Test that handle_error raises in eager mode."""
        ctx = ValidationContext(mode=ValidationMode.EAGER)
        error = SchemaError("Test error")

        with pytest.raises(SchemaError, match="Test error"):
            ctx.handle_error(error)

    def test_handle_error_lazy_mode(self):
        """Test that handle_error collects in lazy mode."""
        result = ValidationResult()
        ctx = ValidationContext(path=["test"], mode=ValidationMode.LAZY, result=result)
        error = SchemaError("Test error")

        ctx.handle_error(error)  # Should not raise

        assert result.has_errors is True
        assert len(result.errors) == 1
        assert result.errors[0] == ("test", error)

    def test_get_errors(self):
        """Test getting collected errors."""
        result = ValidationResult()
        ctx = ValidationContext(mode=ValidationMode.LAZY, result=result)
        error = SchemaError("Test error")
        ctx.handle_error(error)

        errors = ctx.get_errors()
        assert len(errors) == 1
        assert errors[0][1] == error

    def test_has_errors(self):
        """Test checking for errors."""
        result = ValidationResult()
        ctx = ValidationContext(mode=ValidationMode.LAZY, result=result)
        assert ctx.has_errors is False

        ctx.handle_error(SchemaError("Test error"))
        assert ctx.has_errors is True


class TestDataArrayLazyValidation:
    """Test lazy validation for DataArrays."""

    def test_validate_lazy_single_error(self):
        """Test lazy validation with single error."""
        data = np.array([1, 2, 3], dtype=np.float32)
        da = xr.DataArray(data, dims=["x"])

        schema = DataArraySchema(dtype=DTypeSchema(np.int64))
        result = schema.validate(da, mode="lazy")

        assert result.has_errors is True
        assert len(result.errors) == 1
        path, error = result.errors[0]
        assert "dtype" in path
        assert "dtype mismatch" in str(error)

    def test_validate_lazy_multiple_errors(self):
        """Test lazy validation collects multiple errors."""
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        da = xr.DataArray(
            data,
            dims=["y", "x"],
            coords={"x": [0, 1, 2], "y": [0, 1]},
            name="incorrect_name",
        )

        schema = DataArraySchema(
            dtype=DTypeSchema(np.int64),  # Wrong dtype
            dims=DimsSchema(["x", "y"]),  # Wrong dimension order
            name=NameSchema("temperature"),  # Wrong name
        )

        result = schema.validate(da, mode="lazy")

        assert result.has_errors is True
        # Should have at least 3 errors (dtype, dims, name)
        assert len(result.errors) >= 3

        # Check that different error paths are present
        error_paths = [path for path, _ in result.errors]
        assert any("dtype" in path for path in error_paths)
        assert any("dims" in path for path in error_paths)
        assert any("name" in path for path in error_paths)

    def test_validate_lazy_all_pass(self):
        """Test lazy validation when all checks pass."""
        data = np.array([1, 2, 3], dtype=np.int64)
        da = xr.DataArray(data, dims=["x"], name="foo")

        schema = DataArraySchema(
            dtype=DTypeSchema(np.int64),
            dims=DimsSchema(["x"]),
            name=NameSchema("foo"),
        )

        result = schema.validate(da, mode="lazy")

        assert result.has_errors is False
        assert len(result.errors) == 0

    def test_validate_lazy_invalid_input(self):
        """Test that validate_lazy raises for invalid input type."""
        schema = DataArraySchema()
        ds = xr.Dataset()

        with pytest.raises(ValueError, match="Input must be an xarray.DataArray"):
            schema.validate(ds, mode="lazy")

    def test_validate_eager_vs_lazy(self):
        """Test that eager mode stops at first error, lazy collects all."""
        data = np.array([[1, 2, 3]], dtype=np.float32)
        da = xr.DataArray(data, dims=["y", "x"], name="wrong_name")

        schema = DataArraySchema(
            dtype=DTypeSchema(np.int64),
            dims=DimsSchema(["x", "y"]),
            name=NameSchema("correct_name"),
        )

        # Eager mode - should raise on first error
        with pytest.raises(SchemaError):
            schema.validate(da)

        # Lazy mode - should collect all errors
        result = schema.validate(da, mode="lazy")
        assert result.has_errors is True
        assert len(result.errors) >= 2  # Should have multiple errors


class TestDatasetLazyValidation:
    """Test lazy validation for Datasets."""

    def test_validate_lazy_single_data_var_error(self):
        """Test lazy validation with error in single data variable."""
        ds = xr.Dataset(
            data_vars={
                "temperature": (["x", "y"], np.random.random((3, 4)).astype(np.float32))
            },
            coords={"x": [1, 2, 3], "y": [1, 2, 3, 4]},
        )

        schema = DatasetSchema(
            data_vars={
                "temperature": DataArraySchema(
                    dtype=DTypeSchema(np.int64),  # Wrong dtype
                )
            }
        )

        result = schema.validate(ds, mode="lazy")

        assert result.has_errors is True
        assert len(result.errors) >= 1
        # Check error path includes data_vars and variable name
        path, _ = result.errors[0]
        assert "temperature" in path

    def test_validate_lazy_multiple_data_vars_errors(self):
        """Test lazy validation with errors across multiple data variables."""
        temp_data = np.random.random((3, 4)).astype(np.float32)
        pressure_data = np.random.random((3, 4)).astype(np.int32)
        ds = xr.Dataset(
            data_vars={
                "temperature": (["x", "y"], temp_data),
                "pressure": (["x", "y"], pressure_data),
            },
            coords={"x": [1, 2, 3], "y": [1, 2, 3, 4]},
        )

        schema = DatasetSchema(
            data_vars={
                "temperature": DataArraySchema(
                    dtype=DTypeSchema(np.int64),  # Wrong
                    dims=DimsSchema(["y", "x"]),  # Wrong order
                ),
                "pressure": DataArraySchema(
                    dtype=DTypeSchema(np.float64),  # Wrong
                    dims=DimsSchema(["a", "b"]),  # Wrong names
                ),
            }
        )

        result = schema.validate(ds, mode="lazy")

        assert result.has_errors is True
        # Should have errors from both data variables
        error_paths = [path for path, _ in result.errors]
        assert any("temperature" in path for path in error_paths)
        assert any("pressure" in path for path in error_paths)

    def test_validate_lazy_missing_data_var(self):
        """Test lazy validation with missing data variable."""
        ds = xr.Dataset(
            data_vars={
                "temperature": (["x"], np.array([1.0, 2.0, 3.0])),
            }
        )

        schema = DatasetSchema(
            data_vars={
                "temperature": DataArraySchema(dtype=DTypeSchema(np.float64)),
                "pressure": DataArraySchema(dtype=DTypeSchema(np.float64)),
            }
        )

        result = schema.validate(ds, mode="lazy")

        assert result.has_errors is True
        # Should have error about missing 'pressure'
        assert any("pressure" in str(error) for _, error in result.errors)

    def test_validate_lazy_all_pass(self):
        """Test lazy validation when all checks pass."""
        ds = xr.Dataset(data_vars={"temperature": (["x"], np.array([1.0, 2.0, 3.0]))})

        schema = DatasetSchema(
            data_vars={
                "temperature": DataArraySchema(
                    dtype=DTypeSchema(np.float64), dims=DimsSchema(["x"])
                )
            }
        )

        result = schema.validate(ds, mode="lazy")

        assert result.has_errors is False
        assert len(result.errors) == 0


class TestNestedValidationPaths:
    """Test that error paths correctly represent the validation tree."""

    def test_dataarray_coord_error_path(self):
        """Test error path for coordinate validation."""
        da = xr.DataArray(
            [1, 2, 3],
            dims=["x"],
            coords={"x": xr.DataArray([0, 1, 2], dims=["x"], name="x")},
        )

        # Create schema with wrong coord dtype
        from xarray_validate import CoordsSchema

        schema = DataArraySchema(
            coords=CoordsSchema(
                coords={"x": DataArraySchema(dtype=DTypeSchema(np.float64))}
            )
        )

        result = schema.validate(da, mode="lazy")

        if result.has_errors:
            # Error path should include 'coords.x'
            error_paths = [path for path, _ in result.errors]
            assert any("coords" in path and "x" in path for path in error_paths)

    def test_dataset_nested_error_path(self):
        """Test error path for nested Dataset validation."""
        ds = xr.Dataset(
            data_vars={"temp": (["x"], np.array([1, 2, 3], dtype=np.int32))}
        )

        schema = DatasetSchema(
            data_vars={"temp": DataArraySchema(dtype=DTypeSchema(np.float64))}
        )

        result = schema.validate(ds, mode="lazy")

        assert result.has_errors is True
        # Error path should be like 'data_vars.temp.dtype'
        path, _ = result.errors[0]
        assert "data_vars" in path
        assert "temp" in path
        assert "dtype" in path

    def test_error_summary_formatting(self):
        """Test that error summary is properly formatted."""
        data = np.array([1, 2, 3], dtype=np.float32)
        da = xr.DataArray(data, dims=["x"], name="wrong")

        schema = DataArraySchema(
            dtype=DTypeSchema(np.int64),
            name=NameSchema("correct"),
        )

        result = schema.validate(da, mode="lazy")
        summary = result.get_error_summary()

        assert "Validation failed with errors:" in summary
        # Should contain at least one error description
        assert len(summary.split("\n")) > 1
