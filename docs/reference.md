# API Reference

Complete API documentation for xarray-validate.

## Main Schema Classes

The primary schema classes for validating xarray objects:

- [`DataArraySchema`](#xarray_validate.DataArraySchema) - Validate DataArray objects
- [`DatasetSchema`](#xarray_validate.DatasetSchema) - Validate Dataset objects

## Component Schemas

Schema classes for validating specific components:

- [`DTypeSchema`](#xarray_validate.DTypeSchema) - Validate data types
- [`DimsSchema`](#xarray_validate.DimsSchema) - Validate dimensions
- [`ShapeSchema`](#xarray_validate.ShapeSchema) - Validate array shapes
- [`NameSchema`](#xarray_validate.NameSchema) - Validate names
- [`CoordsSchema`](#xarray_validate.CoordsSchema) - Validate coordinates
- [`AttrsSchema`](#xarray_validate.AttrsSchema) - Validate attributes
- [`AttrSchema`](#xarray_validate.AttrSchema) - Validate individual attribute
- [`ChunksSchema`](#xarray_validate.ChunksSchema) - Validate array chunks
- [`ArrayTypeSchema`](#xarray_validate.ArrayTypeSchema) - Validate array types

## Exceptions

- [`SchemaError`](#xarray_validate.SchemaError) - Validation error exception

---

::: xarray_validate
    options:
      show_source: true
      show_root_heading: false
      members: true
      show_bases: true
      inherited_members: true
      heading_level: 2
