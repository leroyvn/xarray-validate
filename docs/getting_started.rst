Getting started
===============

Validating DataArrays
---------------------

A basic DataArray validation schema can be defined as simply as

.. doctest::

    >>> import numpy as np
    >>> from xarray_validate import DataArraySchema

    >>> schema = DataArraySchema(
    ...     dtype=np.int32, name="foo", shape=(4,), dims=["x"]
    ... )

We can then validate a DataArray using its :meth:`.DataArraySchema.validate`
method:

.. doctest::

    >>> import xarray as xr
    >>> da = xr.DataArray(
    ...     np.ones(4, dtype="i4"),
    ...     dims=["x"],
    ...     coords={"x": ("x", np.arange(4)), "y": ("x", np.linspace(0, 1, 4))},
    ...     name="foo",
    ... )
    >>> schema.validate(da)
    None

:meth:`~.DataArraySchema.validate` returns ``None`` if it succeeds.
Validation errors are reported as :class:`.SchemaError`\ s:

.. doctest::

    >>> schema.validate(da.astype("int64"))  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    SchemaError: dtype mismatch: got dtype('int64'), expected dtype('int32')

The :class:`.DataArraySchema` class has many more options, all optional. If not
passed, no validation is performed for that specific part of the DataArray.

The data structures encapsulated within the DataArray can be validated as well.
Each component of the xarray data model has its own validation schema class.
For example:

.. doctest::

    >>> from xarray_validate import CoordsSchema
    >>> schema = DataArraySchema(
    ...     dtype=np.int32,
    ...     name="foo",
    ...     shape=(4,),
    ...     dims=["x"],
    ...     coords=CoordsSchema(
    ...         {"x": DataArraySchema(dtype=np.int64, shape=(4,))}
    ...     )
    ... )
    >>> schema.validate(da)
    None

Validating Datasets
-------------------

Similarly, :class:`xarray.Dataset` instances can be validated using
:class:`.DatasetSchema`. Its ``data_vars`` argument expects a mapping with
variable names as keys and (anything that converts to) :class:`.DataArraySchema`
as values:

.. doctest::

    >>> from xarray_validate import DatasetSchema
    >>> ds = xr.Dataset(
    ...     {
    ...         "x": xr.DataArray(np.arange(4) - 2, dims="x"),
    ...         "foo": xr.DataArray(np.ones(4, dtype="i4"), dims="x"),
    ...         "bar": xr.DataArray(
    ...             np.arange(8, dtype=np.float64).reshape(4, 2), dims=("x", "y")
    ...         ),
    ...     }
    ... )
    >>> schema = DatasetSchema(
    ...     data_vars={
    ...         "foo": DataArraySchema(dtype="<i4", dims=["x"], shape=[4]),
    ...         "bar": DataArraySchema(dtype="<f8", dims=["x", "y"], shape=[4, 2]),
    ...     },
    ...     coords=CoordsSchema(
    ...         {"x": DataArraySchema(dtype="<i8", dims=["x"], shape=(4,))}
    ...     ),
    ... )
    >>> schema.validate(ds)
    None

Eager vs lazy validation mode
-----------------------------

By default, validation errors raise a :class:`SchemaError` eagerly. It is
however possible to perform a lazy Dataset or DataArray validation, during which
errors will be collected and reported after running all subschemas. For example:

.. doctest::
    :options: +NORMALIZE_WHITESPACE

    >>> from xarray_validate import DTypeSchema, DimsSchema, NameSchema
    >>> schema = DataArraySchema(
    ...     dtype=DTypeSchema(np.int64),  # Wrong dtype
    ...     dims=DimsSchema(["x", "y"]),  # Wrong dimension order
    ...     name=NameSchema("temperature"),  # Wrong name
    ... )
    >>> da = xr.DataArray(
    ...     np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
    ...     dims=["y", "x"],
    ...     coords={"x": [0, 1, 2], "y": [0, 1]},
    ...     name="incorrect_name",
    ... )
    >>> schema.validate(da, mode="lazy")
    ValidationResult(errors=[('dtype', SchemaError("dtype mismatch: got dtype('float32'), expected dtype('int64')")),
                             ('name', SchemaError('name mismatch: got incorrect_name, expected temperature')),
                             ('dims', SchemaError('dimension mismatch in axis 0: got y, expected x')),
                             ('dims', SchemaError('dimension mismatch in axis 1: got x, expected y'))])

Loading schemas from serialized data structures
-----------------------------------------------

All component schemas have a :meth:`deserialize` method that allows to
initialize them from basic Python types. The JSON schema for each component maps
to the argument of the respective schema constructor:

.. doctest::

    >>> da = xr.DataArray(
    ...     np.ones(4, dtype="i4"),
    ...     dims=["x"],
    ...     coords={"x": ("x", np.arange(4)), "y": ("x", np.linspace(0, 1, 4))},
    ...     name="foo",
    ... )
    >>> schema = DataArraySchema.deserialize(
    ...     {
    ...         "name": "foo",
    ...         "dtype": "int32",
    ...         "shape": (4,),
    ...         "dims": ["x"],
    ...         "coords": {
    ...             "coords": {
    ...                 "x": {"dtype": "int64", "shape": (4,)},
    ...                 "y": {"dtype": "float64", "shape": (4,)},
    ...             }
    ...         },
    ...     }
    ... )
    >>> schema.validate(da)
    None

This also applies to dataset schemas:

.. doctest::

    >>> ds = xr.Dataset(
    ...     {
    ...         "x": xr.DataArray(np.arange(4) - 2, dims="x"),
    ...         "foo": xr.DataArray(np.ones(4, dtype="i4"), dims="x"),
    ...         "bar": xr.DataArray(
    ...             np.arange(8, dtype=np.float64).reshape(4, 2), dims=("x", "y")
    ...         ),
    ...     }
    ... )
    >>> schema = DatasetSchema.deserialize(
    ...     {
    ...         "data_vars": {
    ...             "foo": {"dtype": "<i4", "dims": ["x"], "shape": [4]},
    ...             "bar": {"dtype": "<f8", "dims": ["x", "y"], "shape": [4, 2]},
    ...         },
    ...         "coords": {
    ...             "coords": {
    ...                 "x": {"dtype": "<i8", "dims": ["x"], "shape": [4]}
    ...             },
    ...         },
    ...     }
    ... )
    >>> schema.validate(ds)
    None

TBD (include YAML)
