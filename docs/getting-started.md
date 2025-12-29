# Getting Started

## Validating DataArrays

A basic DataArray validation schema can be defined as simply as

```python
>>> import numpy as np
>>> from xarray_validate import DataArraySchema

>>> schema = DataArraySchema(
...     dtype=np.int32, name="foo", shape=(4,), dims=["x"]
... )
```

We can then validate a DataArray using its [`DataArraySchema.validate()`](reference.md#xarray_validate.DataArraySchema.validate) method:

```python
>>> import xarray as xr
>>> da = xr.DataArray(
...     np.ones(4, dtype="i4"),
...     dims=["x"],
...     coords={"x": ("x", np.arange(4)), "y": ("x", np.linspace(0, 1, 4))},
...     name="foo",
... )
>>> schema.validate(da)
```

[`validate()`](reference.md#xarray_validate.DataArraySchema.validate) returns `None` if it succeeds.
Validation errors are reported as [`SchemaError`](reference.md#xarray_validate.SchemaError)s:

```python
>>> schema.validate(da.astype("int64"))  # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
...
SchemaError: dtype mismatch: got dtype('int64'), expected dtype('int32')
```

The [`DataArraySchema`](reference.md#xarray_validate.DataArraySchema) class has many more options, all optional. If not passed, no validation is performed for that specific part of the DataArray.

The data structures encapsulated within the DataArray can be validated as well.
Each component of the xarray data model has its own validation schema class.
For example:

```python
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
```

## Validating Datasets

Similarly, [`xarray.Dataset`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html) instances can be validated using [`DatasetSchema`](reference.md#xarray_validate.DatasetSchema). Its `data_vars` argument expects a mapping with variable names as keys and (anything that converts to) [`DataArraySchema`](reference.md#xarray_validate.DataArraySchema) as values:

```python
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
```

## Eager vs lazy validation mode

By default, validation errors raise a `SchemaError` eagerly. It is however possible to perform a lazy Dataset or DataArray validation, during which errors will be collected and reported after running all subschemas. For example:

```python
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
>>> schema.validate(da, mode="lazy")  # doctest: +NORMALIZE_WHITESPACE
ValidationResult(errors=[('dtype', SchemaError("dtype mismatch: got dtype('float32'), expected dtype('int64')")),
                         ('name', SchemaError('name mismatch: got incorrect_name, expected temperature')),
                         ('dims', SchemaError('dimension mismatch in axis 0: got y, expected x')),
                         ('dims', SchemaError('dimension mismatch in axis 1: got x, expected y'))])
```

## Pattern matching for coordinates and data variables

Coordinate and data variable keys in schemas support pattern matching, allowing
you to validate multiple similarly-named items with a single schema definition.
Two pattern types are supported:

**Glob patterns** use wildcards (`*` and `?`) for simple matching:

```python
>>> ds = xr.Dataset(
...     {
...         "x_0": xr.DataArray([1, 2, 3], dims="x"),
...         "x_1": xr.DataArray([4, 5, 6], dims="x"),
...         "x_2": xr.DataArray([7, 8, 9], dims="x"),
...     }
... )
>>> schema = DatasetSchema(
...     data_vars={
...         "x_*": DataArraySchema(dtype=np.int64, dims=["x"], shape=(3,))
...     }
... )
>>> schema.validate(ds)
```

**Regex patterns** use regular expressions enclosed in curly braces for precise matching:

```python
>>> ds = xr.Dataset(
...     {
...         "x_0": xr.DataArray([1, 2, 3], dims="x"),
...         "x_1": xr.DataArray([4, 5, 6], dims="x"),
...         "x_foo": xr.DataArray([7, 8, 9], dims="x"),  # Won't match
...     }
... )
>>> schema = DatasetSchema(
...     data_vars={
...         "{x_\\d+}": DataArraySchema(dtype=np.int64, dims=["x"], shape=(3,))
...     },
...     allow_extra_keys=True,  # Allow x_foo to exist
... )
>>> schema.validate(ds)
```

Pattern matching also works with [`CoordsSchema`](reference.md#xarray_validate.CoordsSchema):

```python
>>> da = xr.DataArray(
...     np.ones((3, 3)),
...     dims=["x", "y"],
...     coords={
...         "x": np.arange(3),
...         "x_label_0": ("x", np.array(["a", "b", "c"], dtype=object)),
...         "x_label_1": ("x", np.array(["d", "e", "f"], dtype=object)),
...     },
... )
>>> schema = DataArraySchema(
...     coords=CoordsSchema(
...         {
...             "x": DataArraySchema(dtype=np.int64),
...             "x_label_*": DataArraySchema(dtype=object),
...         }
...     )
... )
>>> schema.validate(da)
```

**Pattern matching rules:**

- Exact keys take precedence over patterns
- When `require_all_keys=True` (default), only exact keys are required; pattern keys are optional
- When `allow_extra_keys=False`, keys must match either an exact key or a pattern
- Multiple patterns can match the same key; all matching schemas will validate it

!!! tip
    * Learn more about Python's Unix shell-style wildcards in the [`fnmatch`](https://docs.python.org/3/library/fnmatch.html) module documentation.
    * Learn more about Python's regular expressions in the [`re`](https://docs.python.org/3/library/re.html) module documentation.
    * Internally, Unix-style wildcards are converted to regular expressions using the [`fnmatch.translate()`](https://docs.python.org/3/library/fnmatch.html#fnmatch.translate) function.

## Loading schemas from serialized data structures

All component schemas have a `deserialize()` method that allows to initialize them from basic Python types. The JSON schema for each component maps to the argument of the respective schema constructor:

```python
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
```

This also applies to dataset schemas:

```python
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
```

## Loading schemas from YAML files

Schemas can be stored in YAML files for easy version control and sharing.
The `from_yaml()` method, which relies on the `deserialize()` method, is the entry point to load schemas from YAML files.

For DataArrays:

```yaml
# schema.yaml
dtype: float32
name: temperature
shape: [10, 20]
dims: [lat, lon]

coords:
  coords:
    lat:
      dtype: float64
      shape: [10]
    lon:
      dtype: float64
      shape: [20]
```

Load and use the schema:

```python
schema = DataArraySchema.from_yaml("schema.yaml")
schema.validate(my_dataarray)
```

For Datasets:

```yaml
# schema.yaml
data_vars:
  temperature:
    dtype: float32
    dims: [time, lat, lon]
    shape: [12, 180, 360]

  precipitation:
    dtype: float32
    dims: [time, lat, lon]
    shape: [12, 180, 360]

coords:
  coords:
    time:
      dtype: int64
      dims: [time]
      shape: [12]
    lat:
      dtype: float64
      dims: [lat]
      shape: [180]
    lon:
      dtype: float64
      dims: [lon]
      shape: [360]

attrs:
  attrs:
    title: Monthly Climate Data
    institution: Example Climate Center
```

Load and use the Dataset schema:

```python
schema = DatasetSchema.from_yaml("schema.yaml")
schema.validate(my_dataset)
```

!!! note "See also"
    The [examples directory](https://github.com/leroyvn/xarray-validate/tree/main/examples) contains progressive examples demonstrating YAML schema usage.
