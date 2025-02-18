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
)


@pytest.mark.parametrize(
    "component, schema_args, validate, json",
    [
        (
            DTypeSchema,
            np.integer,
            [np.int64, "int", "i8"],
            "<i8",
        ),  # TODO: check Numpy 1
        (DTypeSchema, np.int32, [np.int32, "int32", "i4"], "<i4"),
        (DTypeSchema, "<i8", [np.int64, "int64", "i8"], "<i8"),
        (DimsSchema, ("foo", None), [("foo", "bar"), ("foo", "baz")], ["foo", None]),
        (DimsSchema, ("foo", "bar"), [("foo", "bar")], ["foo", "bar"]),
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
            AttrsSchema,
            {"foo": AttrSchema(value="bar")},
            [{"foo": "bar"}],
            {
                "allow_extra_keys": True,
                "require_all_keys": True,
                "attrs": {"foo": {"type": None, "value": "bar"}},
            },
        ),
        (
            AttrsSchema,
            {"foo": AttrSchema(value=1)},
            [{"foo": 1}],
            {
                "allow_extra_keys": True,
                "require_all_keys": True,
                "attrs": {"foo": {"type": None, "value": 1}},
            },
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
    assert (
        component.deserialize(schema.serialize()).serialize() == json
    ), f"JSON roundtrip of {component} failed"


@pytest.mark.parametrize(
    "type, value, validate, json",
    [
        (str, None, "foo", {"type": str, "value": None}),
        (None, "foo", "foo", {"type": None, "value": "foo"}),
        (str, "foo", "foo", {"type": str, "value": "foo"}),
    ],
)
def test_attr_schema(type, value, validate, json):
    schema = AttrSchema(type=type, value=value)
    schema.validate(validate)
    assert schema.serialize() == json
    # assert isinstance(schema.serialize(), str)


@pytest.mark.parametrize(
    "component, schema_args, value, match",
    [
        (
            DTypeSchema,
            np.integer,
            np.float32,
            r"dtype mismatch: got <class 'numpy.float32'>, expected dtype\('int64'\)",
        ),
        (
            DimsSchema,
            ("foo", "bar"),
            ("foo",),
            "dimension number mismatch: got 1, expected 2",
        ),
        (
            DimsSchema,
            ("foo", "bar"),
            ("foo", "baz"),
            "dimension mismatch in axis 1: got baz, expected bar",
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
