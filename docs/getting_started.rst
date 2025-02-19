Getting started
===============

Installation
------------

Required dependencies:

* Python 3.8 or later
* xarray 2024 or later

Install from PyPI in your virtual environment:

.. code:: shell

    python -m pip install xarray-validate

Extras:

* ``dask``: Validate xarrays with dask arrays.
* ``yaml``: Load schemas from YAML files.

To install all extras, specify the ``all`` extra

.. code:: shell

    python -m pip install "xarray-validate[all]"

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

Validating Datasets
-------------------

TBD

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

Loading schemas from serialized data structures
-----------------------------------------------

All component schemas have a :meth:`deserialize` method that allows to
initialize them from basic Python types. The JSON schema for each component maps
to the argument of the respective schema constructor:

.. doctest::

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

This also applies to dataset schemas:

.. doctest::

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

TBD (include YAML)
