# Changelog

All notable changes to this project are documented in this file.

## 0.0.4 — Upcoming release

### Changed

- Refactored `DTypeSchema` to allow dtype generics and multiple matches
  [1e061507](https://github.com/leroyvn/xarray-validate/commit/1e06150764f7eee88cc9157c7d1a3a46bace4935)

## 0.0.3 — 2025-12-17

### Added

- Lazy validation mode for xarray Datasets and
  DataArrays [#18](https://github.com/leroyvn/xarray-validate/pull/18)
- Support Python 3.14 [#20](https://github.com/leroyvn/xarray-validate/pull/20)

## 0.0.2 — 2025-07-18

### Added

- Pre-commit hook for automatic documentation requirements export
- Taskipy integration for command management
- Support for PEP 735 dependency groups
- `DimsSchema` now support unordered dimension checks
  [#10](https://github.com/leroyvn/xarray-validate/pull/10)

### Fixed

- Fixed dtype conversion for NumPy 2.x compatibility: generic dtypes
  `np.integer` and `np.floating` now convert to `np.int64` and `np.float64`
  respectively

### Infrastructure

- **BREAKING**: Migrated from rye to uv for package management
- Updated CI/CD workflows to use uv
- Updated pre-commit configuration for development workflow
- Updated ReadTheDocs configuration

## 0.0.1 — 2025-02-26

### Added

- Initial release
- `DataArray` and `Dataset` validation
- Basic Python type serialization / deserialization
- Schema construction from existing xarray data
- Support for dtype, dimensions, shape, name, chunks, and attributes validation
- Optional dependencies for dask arrays and YAML schema loading

### Dependencies

- attrs: Core schema definitions
- numpy: Numerical operations and dtype handling
- xarray: Data structure validation target
