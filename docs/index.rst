:hide-toc:

xarray-validate
===============

xarray-validate v\ |release|.

.. image:: https://img.shields.io/pypi/v/xarray-validate?color=blue
   :target: https://pypi.org/project/xarray-validate

.. image:: https://img.shields.io/github/actions/workflow/status/leroyvn/xarray-validate/ci.yml?branch=main
   :target: https://github.com/leroyvn/xarray-validate/actions/workflows/ci.yml

.. image:: https://img.shields.io/readthedocs/xarray-validate
   :target: https://xarray-validate.readthedocs.io

.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json
    :target: https://docs.astral.sh/uv/
    :alt: uv

.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff
    :alt: Ruff

Welcome to the xarray-validate documentation!


Motivation
----------

This is a maintained refactor of
`xarray-schema <https://github.com/xarray-contrib/xarray-schema>`_.
I needed an xarray validation engine for one of my projects. I saw in the
xarray-schema library a good start, but both its maintenance status and the
foreseen integration of its feature set into the much larger Pandera library
seemed uncertain. I therefore decided to fork the project, refactor it and add
the features I was missing.

Features
--------

* ⬆️ DataArray and Dataset validation
* ⬆️ Basic Python type serialization / deserialization
* Construct schema from existing xarray data
* 🚫 JSON roundtrip (not guaranteed to work)

⬆️ Inherited from xarray-schema
🚫 Won't do / won't fix

Installation
------------

Required dependencies:

* Python 3.8 or later
* xarray 2024 or later

Install from PyPI in your virtual environment:

.. code:: shell

    python -m pip install xarray-validate

Available extras:

* ``dask``: Validate xarray containers based on dask arrays.
* ``yaml``: Load schemas from YAML files.

To install all extras:

.. code:: shell

    python -m pip install "xarray-validate[all]"

Development installation:

.. code:: shell

    uv sync --all-groups --all-extras

.. toctree::
    :maxdepth: 2
    :caption: Use
    :hidden:

    getting_started

.. toctree::
    :maxdepth: 2
    :caption: Reference
    :hidden:

    api

.. toctree::
   :caption: Develop
   :hidden:

   GitHub repository <https://github.com/leroyvn/xarray-validate>
