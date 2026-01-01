import dask.array
import numpy as np
import pytest
import xarray as xr

from xarray_validate import (
    ArrayTypeSchema,
    AttrSchema,
    AttrsSchema,
    ChunksSchema,
    CoordsSchema,
    DataArraySchema,
    DimsSchema,
    DTypeSchema,
    NameSchema,
    SchemaError,
    ShapeSchema,
    ValidationContext,
    testing,
)


class TestAttrSchema:
    @pytest.mark.parametrize(
        "kwargs, validate, json",
        [
            (
                {"type": str, "value": None},
                "foo",
                {"type": str, "value": None, "units": None, "units_compatible": None},
            ),
            (
                {"type": None, "value": "foo"},
                "foo",
                {"type": None, "value": "foo", "units": None, "units_compatible": None},
            ),
            (
                {"type": str, "value": "foo"},
                "foo",
                {"type": str, "value": "foo", "units": None, "units_compatible": None},
            ),
        ],
    )
    def test_attr_schema_basic(self, kwargs, validate, json):
        schema = AttrSchema(**kwargs)
        schema.validate(validate)
        assert schema.serialize() == json

    def test_exact_value_match(self):
        """Test that exact matching works for non-pattern values."""
        schema = AttrSchema(value="meters")

        # Exact value matches
        schema.validate("meters")

        # Should not match different value
        with pytest.raises(SchemaError, match="name .* != .*"):
            schema.validate("kilometres")

    def test_glob_pattern_value_matching(self):
        """Test that glob patterns match attribute values."""
        schema = AttrSchema(value="CF-*")

        # Values starting with CF- match
        schema.validate("CF-1.8")
        schema.validate("CF-1.9")

        # Values not starting with CF- do not match
        with pytest.raises(SchemaError, match="does not match pattern"):
            schema.validate("ACDD-1.3")

    def test_regex_pattern_value_matching(self):
        """Test that regex patterns match attribute values."""
        schema = AttrSchema(value=r"{CF-\d+\.\d+}")

        # CF version strings match
        schema.validate("CF-1.8")
        schema.validate("CF-1.10")

        # Invalid formats do not match
        with pytest.raises(SchemaError, match="does not match pattern"):
            schema.validate("CF-1")
        with pytest.raises(SchemaError, match="does not match pattern"):
            schema.validate("ACDD-1.3")

    def test_pattern_value_with_numeric_conversion(self):
        """Test that numeric values are converted to strings for pattern matching."""
        schema = AttrSchema(value=r"{\d+\.\d+}")

        # Numeric values are converted to strings for matching
        schema.validate("1.8")
        schema.validate("2.0")

    def test_pattern_value_in_attrs_schema(self):
        """Test pattern matching for values in AttrsSchema."""
        schema = AttrsSchema.deserialize(
            {
                "Conventions": "CF-*",  # Glob pattern
                "units": "{(metres|kilometres)}",  # Regex pattern
                "comment": None,  # Wildcard (only check if key exists)
            }
        )

        # Validates matching patterns
        schema.validate(
            {"Conventions": "CF-1.8", "units": "metres", "comment": "any value here"}
        )

        # Validates other matching values
        schema.validate(
            {
                "Conventions": "CF-2.0",
                "units": "kilometres",
                "comment": "different value",
            }
        )

        # Fails on non-matching patterns
        ctx = ValidationContext(mode="lazy")
        schema.validate(
            {"Conventions": "ACDD-1.3", "units": "metres", "comment": "test"},
            context=ctx,
        )
        errors = ctx.result.errors
        assert len(errors) == 1
        assert errors[0][0] == "attrs.Conventions"
        assert "does not match pattern" in str(errors[0][1])

    def test_attr_schema_unit_validation_no_pint(self):
        """Test that unit validation fails gracefully without pint."""
        # Create a schema with unit validation
        schema = AttrSchema(units="metre")

        # Mock pint import failure
        import sys

        pint_module = sys.modules.get("pint")
        sys.modules["pint"] = None

        try:
            with pytest.raises(ImportError, match="requires the pint library"):
                schema.validate("metre")
        finally:
            # Restore pint module
            if pint_module:
                sys.modules["pint"] = pint_module
            else:
                sys.modules.pop("pint", None)

    @pytest.mark.parametrize(
        "schema_kwargs, valid_values, invalid_values",
        [
            # Exact unit match - allows different spellings/abbreviations
            (
                {"units": "metre"},
                ["metre", "m", "meter"],  # All equivalent
                ["kilometre", "cm", "foot", "not_a_unit"],
            ),
            (
                {"units": "nanometre"},
                ["nanometer", "nm", "nanometre"],
                ["micrometer", "um", "angstrom", "metre"],
            ),
            (
                {"units": "kelvin"},
                ["kelvin", "K"],
                ["celsius", "fahrenheit", "degC"],
            ),
            (
                {"units": "percent"},
                ["percent", "%"],
                ["dimensionless", "1"],
            ),
        ],
    )
    def test_attr_schema_exact_unit(self, schema_kwargs, valid_values, invalid_values):
        """Test exact unit validation (tolerates different spellings)."""
        pytest.importorskip("pint")

        schema = AttrSchema(**schema_kwargs)

        # Test valid values
        for value in valid_values:
            schema.validate(value)

        # Test invalid values
        for value in invalid_values:
            with pytest.raises(SchemaError, match="(Unit mismatch|Invalid unit)"):
                schema.validate(value)

    @pytest.mark.parametrize(
        "schema_kwargs, valid_values, invalid_values",
        [
            # Compatible units - allows conversions
            (
                {"units_compatible": "metre"},
                [
                    "metre",
                    "m",
                    "kilometre",
                    "km",
                    "centimeter",
                    "cm",
                    "millimeter",
                    "mm",
                    "foot",
                    "mile",
                ],
                ["second", "kelvin", "pascal"],
            ),
            (
                {"units_compatible": "nanometer"},
                [
                    "nanometer",
                    "nm",
                    "micrometer",
                    "um",
                    "angstrom",
                    "metre",
                    "kilometre",
                ],
                ["second", "kelvin"],
            ),
            (
                {"units_compatible": "kelvin"},
                ["kelvin", "K", "celsius", "degC", "fahrenheit"],
                ["metre", "second"],
            ),
            (
                {"units_compatible": "pascal"},
                ["pascal", "Pa", "kPa", "hPa", "bar", "millibar", "atm", "psi"],
                ["metre", "kelvin"],
            ),
            (
                {"units_compatible": "meter / second"},
                ["m/s", "meter/second", "km/h", "kilometre/hour", "mile/hour", "mph"],
                ["metre", "second", "meter**2"],
            ),
            (
                {"units_compatible": "watt / meter**2"},
                ["W/m**2", "watt/meter**2", "W/m^2"],
                ["watt", "metre"],
            ),
        ],
    )
    def test_attr_schema_compatible_units(
        self, schema_kwargs, valid_values, invalid_values
    ):
        """Test compatible unit validation (allows conversions)."""
        pytest.importorskip("pint")

        schema = AttrSchema(**schema_kwargs)

        # Test valid values
        for value in valid_values:
            schema.validate(value)

        # Test invalid values
        for value in invalid_values:
            with pytest.raises(SchemaError, match="not compatible"):
                schema.validate(value)

    def test_attr_schema_unit_validation_non_string(self):
        """Test that unit validation requires string attributes."""
        pytest.importorskip("pint")

        schema = AttrSchema(units="metre")

        with pytest.raises(SchemaError, match="requires attribute to be a string"):
            schema.validate(123)

        with pytest.raises(SchemaError, match="requires attribute to be a string"):
            schema.validate(None)

    def test_attr_schema_invalid_unit_string(self):
        """Test that invalid unit strings are caught."""
        pytest.importorskip("pint")

        schema = AttrSchema(units="metre")

        with pytest.raises(SchemaError, match="Invalid unit"):
            schema.validate("not_a_real_unit")

    def test_attr_schema_serialize_with_units(self):
        """Test serialization includes unit fields."""
        schema = AttrSchema(units="metre")
        result = schema.serialize()
        assert result == {
            "type": None,
            "value": None,
            "units": "metre",
            "units_compatible": None,
        }

        schema = AttrSchema(units_compatible="kelvin")
        result = schema.serialize()
        assert result == {
            "type": None,
            "value": None,
            "units": None,
            "units_compatible": "kelvin",
        }

    def test_attr_schema_deserialize_with_units(self):
        """Test deserialization handles unit fields."""
        schema = AttrSchema.deserialize({"units": "metre"})
        assert schema.units == "metre"
        assert schema.units_compatible is None

        schema = AttrSchema.deserialize({"units_compatible": "kelvin"})
        assert schema.units is None
        assert schema.units_compatible == "kelvin"


class TestAttrsSchema:
    """Tests for AttrsSchema class."""

    @pytest.mark.parametrize(
        "schema_args, validate, json",
        [
            (
                {"foo": AttrSchema(value="bar")},
                [{"foo": "bar"}],
                {
                    "allow_extra_keys": True,
                    "require_all_keys": True,
                    "attrs": {
                        "foo": {
                            "type": None,
                            "value": "bar",
                            "units": None,
                            "units_compatible": None,
                        }
                    },
                },
            ),
            (
                {"foo": AttrSchema(value=1)},
                [{"foo": 1}],
                {
                    "allow_extra_keys": True,
                    "require_all_keys": True,
                    "attrs": {
                        "foo": {
                            "type": None,
                            "value": 1,
                            "units": None,
                            "units_compatible": None,
                        }
                    },
                },
            ),
        ],
    )
    def test_attrs_schema_basic(self, schema_args, validate, json):
        schema = testing.assert_construct(AttrsSchema, schema_args)

        for v in validate:
            schema.validate(v)

        testing.assert_json(schema, json)

    def test_glob_pattern_matching_keys(self):
        """Test that glob patterns match attribute keys."""
        schema = AttrsSchema.deserialize({"valid_*": "pass"})

        # Validates attributes matching the pattern
        schema.validate({"valid_min": "pass", "valid_max": "pass", "other": "ignored"})

        # Fails to validate attributes that do not match the pattern
        with pytest.raises(SchemaError, match="fail"):
            schema.validate({"valid_min": "pass", "valid_max": "fail"})

    def test_regex_pattern_matching_keys(self):
        """Test that regex patterns match attribute keys."""
        schema = AttrsSchema.deserialize({"{valid_(min|max)}": "pass"})

        # Validates attributes matching the regex
        schema.validate({"valid_min": "pass", "valid_max": "pass", "other": "ignore"})

        # Fails to validate attributes that do not match the pattern
        with pytest.raises(SchemaError, match="fail"):
            schema.validate({"valid_min": "pass", "valid_max": "fail"})

    def test_mixed_exact_and_pattern_keys(self):
        """Test mixing exact and pattern keys."""
        schema = AttrsSchema.deserialize(
            {"units": "meters", "valid_*": 0.0, "long_name": "Distance"}
        )

        # Validates with exact and pattern matches
        # Note: All attributes matching valid_* must have value 0.0
        schema.validate(
            {
                "units": "meters",
                "valid_min": 0.0,
                "valid_max": 0.0,
                "long_name": "Distance",
            }
        )

    def test_exact_key_takes_precedence(self):
        """Test that exact keys take precedence over pattern keys."""
        schema = AttrsSchema.deserialize({"valid_min": -10.0, "valid_*": 0.0})

        # valid_min matches exact schema (-10.0), not pattern schema (0.0)
        schema.validate({"valid_min": -10.0, "valid_max": 0.0})

    def test_pattern_with_require_all_keys_false(self):
        """Test pattern matching with optional keys."""
        schema = AttrsSchema.deserialize(
            {
                "attrs": {"valid_*": 0.0},
                "require_all_keys": False,
                "allow_extra_keys": True,
            }
        )

        # Should validate even without pattern matches
        schema.validate({"other_attr": "ignored"})

    def test_pattern_with_allow_extra_keys_false(self):
        """Test pattern matching with strict key checking."""
        schema = AttrsSchema.deserialize(
            {
                "attrs": {"valid_*": 0.0, "units": "meters"},
                "require_all_keys": False,
                "allow_extra_keys": False,
            }
        )

        # Validates when all keys match schema
        # Note: All attributes matching valid_* must have value 0.0
        schema.validate({"valid_min": 0.0, "valid_max": 0.0, "units": "meters"})

        # Raises when there are extra keys
        with pytest.raises(SchemaError, match="attrs has extra keys"):
            schema.validate(
                {"valid_min": 0.0, "units": "meters", "unexpected": "value"}
            )

    def test_multiple_patterns(self):
        """Test multiple pattern keys."""
        schema = AttrsSchema.deserialize({"valid_*": 0.0, r"{flag_\d+}": True})

        # Validates attributes matching different patterns
        # Note: All attributes matching valid_* must have value 0.0
        schema.validate(
            {"valid_min": 0.0, "valid_max": 0.0, "flag_0": True, "flag_1": True}
        )

    def test_pattern_validation_failure(self):
        """Test that pattern validation catches value mismatches."""
        schema = AttrsSchema.deserialize({"valid_*": 0.0})

        # Raises when pattern-matched values don't validate
        with pytest.raises(SchemaError, match="name .* != .*"):
            schema.validate({"valid_min": "wrong_type"})

    def test_empty_pattern_matches_nothing(self):
        """Test that schema with only patterns doesn't require any keys."""
        schema = AttrsSchema.deserialize(
            {"attrs": {"valid_*": 0.0}, "require_all_keys": False}
        )

        # Validates empty attrs when no exact keys are required
        schema.validate({})

    def test_complex_regex_pattern(self):
        """Test complex regex pattern with character classes."""
        schema = AttrsSchema.deserialize({"{[a-z]+_[0-9]{2}}": 100})

        # Should match attributes following the pattern
        schema.validate({"foo_12": 100, "bar_99": 100})

        # Should not match attributes not following the pattern
        with pytest.raises(SchemaError, match="attrs has extra keys"):
            AttrsSchema.deserialize(
                {"attrs": {"{[a-z]+_[0-9]{2}}": 100}, "allow_extra_keys": False}
            ).validate({"foo_1": 100})


class TestDTypeSchema:
    VALIDATION_VALUES = {
        "int64": ["int", np.int64, "int64", "i8"],
        "int32": [np.int32, "int32", "i4"],
        "int16": [np.int16, "int16", "i2"],
    }

    @pytest.mark.parametrize(
        "schema_args, validate, json",
        [
            (np.int64, "int64", "<i8"),
            ("int64", "int64", "<i8"),
            ("<i8", "int64", "<i8"),
            (np.int32, "int32", "<i4"),
            (np.int16, "int16", "<i2"),
        ],
        ids=["integer", "int64", "<i8", "int32", "int16"],
    )
    def test_dtype_schema_basic(self, schema_args, validate, json):
        schema = testing.assert_construct(DTypeSchema, schema_args)

        validate = self.VALIDATION_VALUES[validate]
        for v in validate:
            schema.validate(v)

        testing.assert_json(schema, json)

    def test_dtype_schema_array(self):
        schema = DTypeSchema(["int16", "int32"])
        schema.validate(np.dtype("int16"))
        schema.validate(np.dtype("int32"))
        with pytest.raises(SchemaError):
            schema.validate(np.dtype("int64"))

        schema = DTypeSchema(["integer", "floating"])
        schema.validate(np.dtype("int"))
        schema.validate(np.dtype("float"))
        with pytest.raises(SchemaError):
            schema.validate(np.dtype("bool"))

    def test_dtype_schema_generic(self):
        for dtype in ["float16", "float32", "float64"]:
            DTypeSchema("floating").validate(np.dtype(dtype))

        for dtype in ["int16", "int32", "int64"]:
            DTypeSchema("integer").validate(np.dtype(dtype))


@pytest.mark.parametrize(
    "component, schema_args, validate, json",
    [
        (ShapeSchema, (1, 2, None), [(1, 2, 3), (1, 2, 5)], [1, 2, None]),
        (ShapeSchema, (1, 2, 3), [(1, 2, 3)], [1, 2, 3]),
        (NameSchema, "foo", ["foo"], "foo"),
        (ArrayTypeSchema, np.ndarray, [np.array([1, 2, 3])], "<class 'numpy.ndarray'>"),
        (
            ArrayTypeSchema,
            dask.array.Array,
            [dask.array.zeros(4)],
            "<class 'dask.array.core.Array'>",
        ),
        # schema_args for ChunksSchema include [chunks, dims, shape]
        (ChunksSchema, True, [(((1, 1),), ("x",), (2,))], True),
        (ChunksSchema, {"x": 2}, [(((2, 2),), ("x",), (4,))], {"x": 2}),
        (ChunksSchema, {"x": (2, 2)}, [(((2, 2),), ("x",), (4,))], {"x": [2, 2]}),
        (ChunksSchema, {"x": [2, 2]}, [(((2, 2),), ("x",), (4,))], {"x": [2, 2]}),
        (ChunksSchema, {"x": 4}, [(((4,),), ("x",), (4,))], {"x": 4}),
        (ChunksSchema, {"x": -1}, [(((4,),), ("x",), (4,))], {"x": -1}),
        (
            ChunksSchema,
            {"x": (1, 2, 1)},
            [(((1, 2, 1),), ("x",), (4,))],
            {"x": [1, 2, 1]},
        ),
        (
            ChunksSchema,
            {"x": 2, "y": -1},
            [(((2, 2), (10,)), ("x", "y"), (4, 10))],
            {"x": 2, "y": -1},
        ),
        (
            CoordsSchema,
            {"x": DataArraySchema(name="x")},
            [{"x": xr.DataArray([0, 1], name="x")}],
            {
                "coords": {"x": {"name": "x"}},
                "allow_extra_keys": True,
                "require_all_keys": True,
            },
        ),
    ],
)
def test_component_schema(component, schema_args, validate, json):
    """
    Generic tests for all schema components.
    """
    # Initialization
    try:
        schema = component(schema_args)
    except TypeError:
        print(f"init of {component} from {schema_args} failed")
        raise

    # Validation
    for v in validate:
        if component in [ChunksSchema]:  # special case construction
            schema.validate(*v)
        else:
            schema.validate(v)

    # JSON checks
    assert schema.serialize() == json, f"JSON export of {component} failed"

    # JSON roundtrip
    assert component.deserialize(schema.serialize()).serialize() == json, (
        f"JSON roundtrip of {component} failed"
    )


@pytest.mark.parametrize(
    "dims, ordered, validate, json",
    [
        (
            ("foo", None),
            None,
            [("foo", "bar"), ("foo", "baz")],
            ["foo", None],
        ),
        (
            ("foo", "bar"),
            None,
            [("foo", "bar")],
            ["foo", "bar"],
        ),
        (
            ("foo", "bar"),
            False,
            [("foo", "bar"), ("bar", "foo")],
            {"dims": ["foo", "bar"], "ordered": False},
        ),
        (
            ("foo", None),
            False,
            [("foo", "bar"), ("bar", "foo")],
            {"dims": ["foo", None], "ordered": False},
        ),
    ],
)
def test_dims_schema(dims, ordered, validate, json):
    # Initialization
    kwargs = {}
    if ordered is not None:
        kwargs["ordered"] = ordered

    schema = DimsSchema(dims, **kwargs)

    # Validation
    for v in validate:
        schema.validate(v)

    # JSON checks
    assert schema.serialize() == json

    # JSON roundtrip
    assert DimsSchema.deserialize(schema.serialize()).serialize() == json, (
        "JSON roundtrip of DimsSchema failed"
    )


@pytest.mark.parametrize(
    "component, schema_args, value, match",
    [
        (
            DTypeSchema,
            np.integer,
            np.float32,
            "dtype mismatch: got <class 'numpy.float32'>, expected "
            "<class 'numpy.integer'>",
        ),
        (
            ShapeSchema,
            (1, 2, None),
            (1, 2),
            "dimension count mismatch: got 2, expected 3",
        ),
        (
            ShapeSchema,
            (1, 4, 4),
            (1, 3, 4),
            "shape mismatch in axis 1: got 3, expected 4",
        ),
        (NameSchema, "foo", "bar", "name mismatch: got bar, expected foo"),
        (
            ArrayTypeSchema,
            np.ndarray,
            "bar",
            "array type mismatch: got <class 'str'>, expected <class 'numpy.ndarray'>",
        ),
        # schema_args for ChunksSchema include [chunks, dims, shape]
        (ChunksSchema, {"x": 3}, (((2, 2),), ("x",), (4,)), r"chunk mismatch*."),
        (ChunksSchema, {"x": (2, 1)}, (((2, 2),), ("x",), (4,)), r"chunk mismatch.*"),
        (
            ChunksSchema,
            {"x": (2, 1)},
            (None, ("x",), (4,)),
            r".*expected array to be chunked.*",
        ),
        (ChunksSchema, True, (None, ("x",), (4,)), r".*expected array to be chunked.*"),
        (
            ChunksSchema,
            False,
            (((2, 2),), ("x",), (4,)),
            r".*expected unchunked array but it is chunked*",
        ),
        (ChunksSchema, {"x": -1}, (((1, 2, 1),), ("x",), (4,)), r"chunk mismatch.*"),
        (ChunksSchema, {"x": 2}, (((2, 3, 2),), ("x",), (7,)), r"chunk mismatch.*"),
        (ChunksSchema, {"x": 2}, (((2, 2, 3),), ("x",), (7,)), r"chunk mismatch.*"),
        (
            ChunksSchema,
            {"x": 2, "y": -1},
            (((2, 2), (5, 5)), ("x", "y"), (4, 10)),
            r"chunk mismatch.*",
        ),
    ],
)
def test_component_raises_schema_error(component, schema_args, value, match):
    schema = component(schema_args)
    with pytest.raises(SchemaError, match=match):
        if component in [ChunksSchema]:  # special case construction
            schema.validate(*value)
        else:
            schema.validate(value)


@pytest.mark.parametrize(
    "dims, ordered, value, match",
    [
        (
            ("foo", "bar"),
            None,
            ("foo",),
            "dimension number mismatch: got 1, expected 2",
        ),
        (
            ("foo", "bar"),
            None,
            ("foo", "baz"),
            "dimension mismatch in axis 1: got baz, expected bar",
        ),
        (
            ("foo", "bar"),
            False,
            ("foo", "baz"),
            "dimension mismatch: expected bar is missing from actual dimension list "
            r"\('foo', 'baz'\)",
        ),
    ],
)
def test_dims_raises_schema_error(dims, ordered, value, match):
    kwargs = {}
    if ordered is not None:
        kwargs["ordered"] = ordered
    schema = DimsSchema(dims, **kwargs)

    with pytest.raises(SchemaError, match=match):
        schema.validate(value)


def test_chunks_schema_raises_for_invalid_chunks():
    with pytest.raises(
        TypeError,
        match=r"'chunks' must be \(<class 'bool'>, <class 'dict'>\) "
        r"\(got 2 that is a <class 'int'>\).",
    ):
        ChunksSchema(chunks=2)


def test_unknown_array_type_raises():
    with pytest.raises(
        TypeError,
        match=r"'array_type' must be <class 'type'> "
        r"\(got 'foo.array' that is a <class 'str'>\).",
    ):
        ArrayTypeSchema.deserialize("foo.array")
