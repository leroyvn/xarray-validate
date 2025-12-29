# Getting Started

## Validating DataArrays

A basic DataArray validation schema can be defined as simply as

```python
>>> import numpy as np
>>> import xarray_validate as xv

>>> schema = xv.DataArraySchema(
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
>>> schema = xv.DataArraySchema(
...     dtype=np.int32,
...     name="foo",
...     shape=(4,),
...     dims=["x"],
...     coords=xv.CoordsSchema(
...         {"x": xv.DataArraySchema(dtype=np.int64, shape=(4,))}
...     )
... )
>>> schema.validate(da)
```

## Validating Datasets

Similarly, [`xarray.Dataset`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html) instances can be validated using [`DatasetSchema`](reference.md#xarray_validate.DatasetSchema). Its `data_vars` argument expects a mapping with variable names as keys and (anything that converts to) [`DataArraySchema`](reference.md#xarray_validate.DataArraySchema) as values:

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
>>> schema = xv.DatasetSchema(
...     data_vars={
...         "foo": xv.DataArraySchema(dtype="<i4", dims=["x"], shape=[4]),
...         "bar": xv.DataArraySchema(dtype="<f8", dims=["x", "y"], shape=[4, 2]),
...     },
...     coords=xv.CoordsSchema(
...         {"x": xv.DataArraySchema(dtype="<i8", dims=["x"], shape=(4,))}
...     ),
... )
>>> schema.validate(ds)
```

## Constructing schemas from existing data

Instead of manually defining schemas, you can automatically generate them from
existing xarray objects using the `from_dataarray()` and `from_dataset()`
factory methods. This is mainly useful for creating a baseline schema from a
reference data file. Since this feature leverages the serialization
infrastructure, it is likely that the produced schema will contain too much
validation details for the desired used case.

For DataArrays:

```python
>>> da = xr.DataArray(
...     np.ones(4, dtype="i4"),
...     dims=["x"],
...     coords={"x": ("x", np.arange(4)), "y": ("x", np.linspace(0, 1, 4))},
...     name="foo",
... )
>>> schema = xv.DataArraySchema.from_dataarray(da)
>>> schema.validate(da)  # Validates successfully
```

The generated schema captures all structural properties:

```python
>>> schema.dtype
DTypeSchema(dtype=dtype('int32'))
>>> schema.name
NameSchema(name='foo')
>>> schema.dims
DimsSchema(dims=('x',), ordered=True)
```

For Datasets:

```python
>>> ds = xr.Dataset(
...     {
...         "foo": xr.DataArray(np.ones(4, dtype="i4"), dims="x"),
...         "bar": xr.DataArray(
...             np.arange(8, dtype=np.float64).reshape(4, 2), dims=("x", "y")
...         ),
...     },
...     coords={"x": np.arange(4)},
... )
>>> schema = xv.DatasetSchema.from_dataset(ds)
>>> schema.validate(ds)  # Validates successfully
```

The generated schemas can be serialized for storage and reuse:

```python
>>> serialized = schema.serialize()
>>> reconstructed_schema = xv.DatasetSchema.deserialize(serialized)
>>> reconstructed_schema.validate(ds)  # Works with the original dataset
```

## Eager vs lazy validation mode

By default, validation errors raise a `SchemaError` eagerly. It is however possible to perform a lazy Dataset or DataArray validation, during which errors will be collected and reported after running all subschemas. For example:

```python
>>> schema = xv.DataArraySchema(
...     dtype=xv.DTypeSchema(np.int64),  # Wrong dtype
...     dims=xv.DimsSchema(["x", "y"]),  # Wrong dimension order
...     name=xv.NameSchema("temperature"),  # Wrong name
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

## Pattern matching for coordinates, data variables and attributes

Coordinate and data variable keys in schemas support pattern matching, allowing
you to validate multiple similarly-named items with a single schema definition.
This also applies to attribute keys and string values. Two pattern types are
supported:

* **Glob patterns** use wildcards (`*` and `?`) for simple matching:

  ```python
  >>> ds = xr.Dataset(
  ...     {
  ...         "x_0": xr.DataArray([1, 2, 3], dims="x"),
  ...         "x_1": xr.DataArray([4, 5, 6], dims="x"),
  ...         "x_2": xr.DataArray([7, 8, 9], dims="x"),
  ...     }
  ... )
  >>> schema = xv.DatasetSchema(
  ...     data_vars={
  ...         "x_*": xv.DataArraySchema(dtype=np.int64, dims=["x"], shape=(3,))
  ...     }
  ... )
  >>> schema.validate(ds)
  ```

* **Regex patterns** use regular expressions enclosed in curly braces for precise matching:

  ```python
  >>> ds = xr.Dataset(
  ...     {
  ...         "x_0": xr.DataArray([1, 2, 3], dims="x"),
  ...         "x_1": xr.DataArray([4, 5, 6], dims="x"),
  ...         "x_foo": xr.DataArray([7, 8, 9], dims="x"),  # Won't match
  ...     }
  ... )
  >>> schema = xv.DatasetSchema(
  ...     data_vars={
  ...         "{x_\\d+}": xv.DataArraySchema(dtype=np.int64, dims=["x"], shape=(3,))
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
>>> schema = xv.DataArraySchema(
...     coords=xv.CoordsSchema(
...         {
...             "x": xv.DataArraySchema(dtype=np.int64),
...             "x_label_*": xv.DataArraySchema(dtype=object),
...         }
...     )
... )
>>> schema.validate(da)
```

!!! info "Pattern matching rules"
    - Exact keys take precedence over patterns
    - When `require_all_keys=True` (default), only exact keys are required; pattern keys are optional
    - When `allow_extra_keys=False`, keys must match either an exact key or a pattern
    - Multiple patterns can match the same key; all matching schemas will validate it

!!! tip
    * Learn more about Python's wildcards in the [`fnmatch`](https://docs.python.org/3/library/fnmatch.html) module documentation.
    * Learn more about Python's regular expressions in the [`re`](https://docs.python.org/3/library/re.html) module documentation.
    * Internally, wildcards are converted to regular expressions using the [`fnmatch.translate()`](https://docs.python.org/3/library/fnmatch.html#fnmatch.translate) function.

## Unit validation

Attributes are often used to specify the units in which numerical values are expressed.
Pattern matching can be used for basic cases, but that requires being exhaustive and potentially
good at using regular expression. To simplify unit value checks, the
[`AttrSchema`](reference.md#xarray_validate.AttrSchema) class supports two specific arguments that trigger unit
validation using the [Pint library](https://github.com/hgrecco/pint):

* `units` expects specific units but tolerates many kinds of abbreviations or alternative spelling (e.g. all of `"metre"`, `"meter"`, and `"m"` will be accepted):

  ```python
  >>> da_meter = xr.DataArray([1.0], attrs={"units": "meter"})
  >>> da_m = xr.DataArray([1.0], attrs={"units": "m"})
  >>> schema = xv.DataArraySchema(
  ...     attrs=xv.AttrsSchema({"units": xv.AttrSchema(units="metre")})
  ... )
  >>> schema.validate(da_meter)  # All equivalent to "metre"
  >>> schema.validate(da_m)
  ```

  It rejects all other units:

  ```python
  >>> da_km = xr.DataArray([1.0], attrs={"units": "km"})
  >>> schema.validate(da_km)  # doctest: +IGNORE_EXCEPTION_DETAIL
  Traceback (most recent call last):
  ...
  SchemaError: Unit mismatch: got kilometre, expected metre
  ```

* `units_compatible` is more flexible and accepts all units with the same dimensionality (e.g. all of `"m"`, `"mm"`, and `"km"` will be accepted).

  ```python
  >>> schema = xv.DataArraySchema(
  ...     attrs=xv.AttrsSchema({"units": xv.AttrSchema(units_compatible="metre")})
  ... )
  >>> da_km = xr.DataArray([1.0], attrs={"units": "km"})
  >>> da_mm = xr.DataArray([1.0], attrs={"units": "mm"})
  >>> schema.validate(da_km)
  >>> schema.validate(da_mm)
  ```

## Loading schemas from serialized data structures

All component schemas have a `deserialize()` method that allows to initialize them from basic Python types. The JSON schema for each component maps to the argument of the respective schema constructor:

```python
>>> da = xr.DataArray(
...     np.ones(4, dtype="i4"),
...     dims=["x"],
...     coords={"x": ("x", np.arange(4)), "y": ("x", np.linspace(0, 1, 4))},
...     name="foo",
... )
>>> schema = xv.DataArraySchema.deserialize(
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
>>> schema = xv.DatasetSchema.deserialize(
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
The `from_yaml()` method, which relies on the `deserialize()` method, is the
entry point to load schemas from YAML files.

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
schema = xv.DataArraySchema.from_yaml("schema.yaml")
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
schema = xv.DatasetSchema.from_yaml("schema.yaml")
schema.validate(my_dataset)
```

!!! note "See also"
    The [examples directory](https://github.com/leroyvn/xarray-validate/tree/main/examples) contains progressive examples demonstrating YAML schema usage.
