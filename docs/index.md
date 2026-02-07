# xarray-validate

[![PyPI Version](https://img.shields.io/pypi/v/xarray-validate?color=blue)](https://pypi.org/project/xarray-validate)
[![CI Status](https://img.shields.io/github/actions/workflow/status/leroyvn/xarray-validate/ci.yml?branch=main)](https://github.com/leroyvn/xarray-validate/actions/workflows/ci.yml)
[![ReadTheDocs](https://img.shields.io/readthedocs/xarray-validate)](https://xarray-validate.readthedocs.io)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://docs.astral.sh/uv/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Welcome to the xarray-validate documentation!

## Motivation

This is a maintained refactor of [xarray-schema](https://github.com/xarray-contrib/xarray-schema).
I needed an xarray validation engine for one of my projects. I saw in the
xarray-schema library a good start, but both its maintenance status and the
foreseen integration of its feature set into the much larger Pandera library
seemed uncertain. I therefore decided to fork the project, refactor it and add
the features I was missing.

## Features

* ⬆️ DataArray and Dataset validation
* ⬆️ Basic Python type serialization / deserialization
* Construct schema from existing xarray data
* 🚫 JSON roundtrip (not guaranteed to work)

⬆️ Inherited from xarray-schema
🚫 Won't do / won't fix

## Installation

Required dependencies:

* Python 3.8 or later
* xarray 2024 or later

Install from PyPI in your virtual environment:

```shell
python -m pip install xarray-validate
```

Available extras:

* `dask`: Validate xarray containers based on dask arrays.
* `yaml`: Load schemas from YAML files.

To install all extras:

```shell
python -m pip install "xarray-validate[all]"
```

Development installation:

```shell
uv sync --all-groups --all-extras
```
