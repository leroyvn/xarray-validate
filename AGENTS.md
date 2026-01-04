# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

To use it with Claude Code, simply run the following command in a terminal:

```bash
echo "@AGENTS.md" > CLAUDE.md
```

## Development Commands

This project uses uv as the package manager and taskipy for task automation.

**Setup:**
```bash
uv sync --dev --all-extras
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
- `DTypeSchema`: Validates NumPy data types (supports dtype generics and multiple matches)
- `DimsSchema`: Validates dimensions (with ordered/unordered support via `ordered` parameter)
- `ShapeSchema`: Validates array shapes (supports wildcards with None)
- `NameSchema`: Validates names
- `ChunksSchema`: Validates dask chunks (bool or dict specification)
- `ArrayTypeSchema`: Validates underlying array types
- `AttrSchema`/`AttrsSchema`: Validates attributes (supports pattern matching for keys and values)
- `UnitsSchema`: Validates physical units using Pint (requires pint optional dependency)

**Pattern Matching Utilities (`_match.py`):**
- `_is_regex_pattern()`: Checks if a key is a regex pattern (enclosed in curly braces)
- `_is_glob_pattern()`: Checks if a key is a glob pattern (contains * or ?)
- `_is_pattern_key()`: Checks if a key is any kind of pattern
- `_pattern_to_regex()`: Converts glob or regex patterns to compiled regex objects
- Used for pattern-based matching in coordinate keys, data variable keys, and attribute keys/values

**Unit Validation Utilities (`units.py`):**
- `set_registry()`: Set the global Pint unit registry
- `get_registry()`: Get the current Pint unit registry
- `parse()`: Parse unit strings with error handling
- Provides utilities for validating physical units in attributes

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
- `tests/test_components.py`: Tests for validation components (including unit validation)
- `tests/test_dataarray.py`: Tests for DataArray schema validation
- `tests/test_dataset.py`: Tests for Dataset schema validation
- `tests/test_lazy_validation.py`: Tests for lazy validation mode
- `tests/test_match.py`: Tests for pattern matching utilities
- `tests/test_yaml_examples.py`: Tests for YAML schema loading
- `conftest.py`: Shared test fixtures (located in project root)

The project uses pytest with xdoctest for docstring testing and coverage reporting.

## Dependencies and Compatibility

**Core dependencies:** attrs, numpy, xarray
**Optional dependencies:**
- dask (for chunk validation)
- ruamel-yaml (for YAML support)
- pint (for unit validation)

**Python support:** 3.8 through 3.14
**Build system:** hatchling with uv as package manager

## Code Style

- Uses Ruff for linting and formatting (configured in pyproject.toml)
- Follows attrs/dataclass patterns for schema definition
- Type hints throughout codebase
- Imports follow isort configuration with relative imports ordered closest-to-furthest

## Code Organization

**Module Structure:**
- `base.py`: Base classes and core abstractions
- `components.py`: Individual validation components (including UnitsSchema)
- `_match.py`: Pattern matching utilities (used by dataarray.py, dataset.py, and components.py)
- `units.py`: Unit validation utilities using Pint
- `dataarray.py`: DataArray and Coordinates schemas
- `dataset.py`: Dataset schema
- `types.py`: Type definitions
- `testing.py`: Testing utilities
- `converters.py`: Type converters for attrs

**Key Principles:**
- Shared utilities are factored into dedicated modules (e.g., `_match.py`)
- Private modules prefixed with underscore are for internal use only
- Pattern matching logic is centralized to avoid code duplication
