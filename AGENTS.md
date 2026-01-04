# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

This project uses uv as the package manager and taskipy for task automation.

**Setup:**
```bash
uv sync --locked --all-extras --dev
```

**Testing:**
```bash
uv run task test               # Run all tests
uv run task test-cov           # Run tests with coverage
uv run task test-cov-report    # Generate HTML coverage report
```

**Code Quality:**
```bash
uv run ruff check .            # Check linting
uv run ruff format .           # Format code
```

**Documentation:**
```bash
uv run task docs               # Build docs
uv run task docs-serve         # Serve docs with auto-reload
uv run task docs-clean         # Clean docs build
```

**Running single tests:**
```bash
uv run pytest tests/test_dataarray.py::TestDataArraySchema::test_validate_dtype
```

## Architecture Overview

xarray-validate is a lightweight validation library for xarray DataArrays and Datasets, refactored from xarray-schema. The architecture follows a component-based validation pattern:

### Core Components

**Base Classes (`base.py`):**
- `BaseSchema`: Abstract base for all schema classes with serialize/deserialize/validate pattern
- `SchemaError`: Custom exception for validation failures

**Validation Components (`components.py`):**
- `DTypeSchema`: Validates NumPy data types
- `DimsSchema`: Validates dimensions (with ordered/unordered support via `ordered` parameter)
- `ShapeSchema`: Validates array shapes (supports wildcards with None)
- `NameSchema`: Validates names
- `ChunksSchema`: Validates dask chunks (bool or dict specification)
- `ArrayTypeSchema`: Validates underlying array types
- `AttrSchema`/`AttrsSchema`: Validates attributes

**Pattern Matching Utilities (`_match.py`):**
- `_is_regex_pattern()`: Checks if a key is a regex pattern (enclosed in curly braces)
- `_is_glob_pattern()`: Checks if a key is a glob pattern (contains * or ?)
- `_is_pattern_key()`: Checks if a key is any kind of pattern
- `_pattern_to_regex()`: Converts glob or regex patterns to compiled regex objects
- Used for pattern-based matching in coordinate and data variable keys

**High-Level Schemas:**
- `DataArraySchema` (`dataarray.py`): Combines all validation components for xarray.DataArray objects
- `CoordsSchema` (`dataarray.py`): Validates coordinate collections (supports glob and regex patterns)
- `DatasetSchema` (`dataset.py`): Validates xarray.Dataset objects (supports glob and regex patterns for data_vars and coords)

### Key Design Patterns

**Validation Pattern:** All schemas implement `validate(obj)` that raises `SchemaError` on failure.

**Conversion Pattern:** All schemas support automatic conversion via `convert()` class method and attrs converters.

**Serialization:** Schemas can serialize to/from basic Python types for JSON/YAML persistence.

**Factory Methods:** `DataArraySchema.from_dataarray()` and `DatasetSchema.from_dataset()` create schemas from existing xarray objects.

**Pattern Matching:** Coordinate and data variable keys support two pattern types:
- Glob patterns: `'x_*'` matches `x_0`, `x_1`, `x_foo`, etc.
- Regex patterns: `'{x_\\d+}'` matches `x_0`, `x_1`, but not `x_foo` (enclosed in curly braces)

## Testing Structure

Tests are organized by component:
- `tests/test_components.py`: Tests for validation components
- `tests/test_dataarray.py`: Tests for DataArray schema validation
- `tests/test_dataset.py`: Tests for Dataset schema validation
- `tests/conftest.py`: Shared test fixtures

The project targets >90% test coverage and uses pytest with coverage reporting.

## Dependencies and Compatibility

**Core dependencies:** attrs, numpy, xarray
**Optional dependencies:** dask (for chunk validation), ruamel-yaml (for YAML support)
**Python support:** 3.8 through 3.13
**Build system:** hatchling with uv as package manager

## Code Style

- Uses Ruff for linting and formatting (configured in pyproject.toml)
- Follows attrs/dataclass patterns for schema definition
- Type hints throughout codebase
- Imports follow isort configuration with relative imports ordered closest-to-furthest

## Code Organization

**Module Structure:**
- `base.py`: Base classes and core abstractions
- `components.py`: Individual validation components
- `_match.py`: Pattern matching utilities (shared by dataarray.py and dataset.py)
- `dataarray.py`: DataArray and Coordinates schemas
- `dataset.py`: Dataset schema
- `types.py`: Type definitions

**Key Principles:**
- Shared utilities are factored into dedicated modules (e.g., `_match.py`)
- Private modules prefixed with underscore are for internal use only
- Pattern matching logic is centralized to avoid code duplication
