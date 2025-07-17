# Contributing to xarray-validate

Thank you for your interest in contributing to xarray-validate! This guide will
help you get started with development and understand the project's workflow.

## Getting started

### Prerequisites

- Python 3.8 or later
- [uv](https://docs.astral.sh/uv/) package manager
- Git

### Development setup

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/yourusername/xarray-validate.git
   cd xarray-validate
   ```

2. **Install the project in development mode**:
   ```bash
   uv sync --locked --all-extras --dev
   ```

   This will install all dependencies including development tools and optional
   extras.

3. **Verify the installation**:
   ```bash
   uv run python -c "import xarray_validate; print('Installation successful!')"
   ```

## Development workflow

### Code style and quality

This project uses several tools to maintain code quality:

- **Ruff**: For linting and code formatting
- **pytest**: For testing
- **Coverage**: For test coverage reporting

### Running Tests

Run the full test suite:
```bash
uv run task test
```

Run tests with coverage:
```bash
uv run task test-cov
```

Generate HTML coverage report:
```bash
uv run task test-cov-report
```

The coverage report will be available in `reports/coverage/html/index.html`.

Our CI runs the test suite for all supported Python versions and reports
coverage.

### Code formatting and linting

The project uses Ruff for code formatting and linting. The configuration is in
`pyproject.toml`.

To check your code:
```bash
uv run ruff check .
```

To format your code:
```bash
uv run ruff format .
```

Pre-commit hooks are set up and will automatically format modified files upon
committing.

### Building Documentation

The documentation is built using Sphinx. Available tasks:

```bash
# Build documentation
uv run task docs

# Clean documentation build
uv run task docs-clean

# Serve documentation with auto-reload
uv run task docs-serve

# Update documentation requirements
uv run task docs-lock
```

After building, the documentation will be available in `docs/_build/html/`.

A pre-commit hook automatically updates docs requirements when necessary.

## Project Structure

```
xarray-validate/
├── src/xarray_validate/          # Main package source code
│   ├── __init__.py              # Package initialization
│   ├── base.py                  # Base validation components
│   ├── components.py            # Validation components
│   ├── converters.py            # Type converters
│   ├── dataarray.py             # DataArray validation
│   ├── dataset.py               # Dataset validation
│   └── types.py                 # Type definitions
├── tests/                       # Test suite
│   ├── conftest.py              # Test configuration
│   ├── test_components.py       # Component tests
│   ├── test_dataarray.py        # DataArray tests
│   └── test_dataset.py          # Dataset tests
├── docs/                        # Documentation source
│   ├── conf.py                  # Sphinx configuration
│   ├── index.rst                # Documentation index
│   ├── getting_started.rst      # Getting started guide
│   └── api.rst                  # API reference
├── pyproject.toml              # Project configuration
└── README.md                   # Project README
```

## Making contributions

### Types of contributions

We welcome various types of contributions:

- **Bug fixes**: Fix issues in the codebase
- **Feature additions**: Add new validation capabilities
- **Documentation**: Improve or add documentation
- **Tests**: Add or improve test coverage
- **Performance**: Optimize existing code

### Contribution process

1. **Create an issue** (optional but recommended):
   - For bug reports, include a minimal reproducible example
   - For feature requests, describe the use case and expected behavior

2. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

3. **Make your changes**:
   - Write clear, concise code
   - Add tests for new functionality
   - Update documentation if needed
   - Ensure all tests pass

4. **Test your changes**:
   ```bash
   uv run task test
   uv run ruff check .
   ```

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add clear description of changes"
   ```

6. **Push and create a pull request**:
   ```bash
   git push origin your-branch-name
   ```

### Pull request guidelines

- **Title**: Use a clear, descriptive title
- **Description**: Include:
  - What changes were made
  - Why the changes were necessary
  - Any breaking changes
  - Related issues (use "Fixes #123" to auto-close issues)
- **Tests**: Ensure all tests pass and add tests for new functionality
- **Documentation**: Update documentation for user-facing changes
- **Code Quality**: Follow the project's code style and conventions

### Writing tests

- Place tests in the `tests/` directory
- Use descriptive test names that explain what is being tested
- Include both positive and negative test cases
- Test edge cases and error conditions
- Aim for high test coverage (current target: >90%)

### Documentation guidelines

- Use clear, concise language
- Include code examples where helpful
- Update the API documentation for new functions/classes
- Add docstrings to all public functions and classes
- Follow the existing documentation style

## Release process

This project uses semantic versioning (SemVer). Releases are automated through
GitHub Actions when tags are pushed.

Version format: `MAJOR.MINOR.PATCH`
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

## Getting help

If you need help:

1. Check the existing documentation
2. Search existing issues on GitHub
3. Create a new issue with your question
4. Join discussions in existing issues

## Code of conduct

Please be respectful and constructive in all interactions. We aim to create a
welcoming environment for all contributors.

## License

By contributing to xarray-validate, you agree that your contributions will be
licensed under the MIT License.

## Acknowledgments

This project builds upon the work of the xarray-schema project. We acknowledge
and appreciate the contributions of the original creators and maintainers.
