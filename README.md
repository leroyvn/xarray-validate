# xarray-validate

[![PyPI version](https://img.shields.io/pypi/v/xarray-validate?color=blue)](https://pypi.org/project/xarray-validate)

[![GitHub Workflow Status (branch)](https://img.shields.io/github/actions/workflow/status/leroyvn/xarray-validate/ci.yml?branch=main)](https://github.com/leroyvn/pinttrs/actions/workflows/ci.yml)
[![Documentation Status](https://img.shields.io/readthedocs/xarray-validate)](https://xarray-validate.readthedocs.io)

[![Rye](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/mitsuhiko/rye/main/artwork/badge.json)](https://rye-up.com)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Motivation

*This library is a fork of
[xarray-schema](https://github.com/xarray-contrib/xarray-schema).*

I needed an xarray validation engine for one of my projects. The only solid
contender I could see was xarray-schema, but its maintenance seems uncertain and
its integration into the much larger Pandera project was not progressing quickly
enough for me. I therefore decided to fork the project, refactor it and add the
features I was missing.

## Features

* DataArray and Dataset validation ⬆️
* Basic Python type serialization / deserialization ⬆️
* Construct schema from existing xarray data
* ~~JSON roundtrip~~ (not guaranteed to work) 🚫

⬆️ Inherited from xarray-schema
🚫 Won't do / won't fix

## License

Pinttrs is distributed under the terms of the
[MIT license](https://choosealicense.com/licenses/mit/).

## About

xarray-validate is maintained by [Vincent Leroy](https://github.com/leroyvn).

The xarray-validate maintainers acknowledge the work of the xarray-schema
project creators and maintainers.
