# xarray-validate

This library is a port of [xarray-schema](https://github.com/xarray-contrib/xarray-schema)
to the attrs library to reduce boilerplate and make schema validation and
conversion more comprehensive.

My goals are as follows:

* To get to the essentials: Having an xarray validation library that supports my
  use case. Since I need to be able to maintain it, I need a codebase I enjoy
  working with, and getting rid of class definition boilerplate is a priority.
* To add a custom rule system to perform advanced validation tasks (e.g.
  verifying units with the Pint library).
* To simplify the tooling and CI/CD processes. Let's not overcomplicate things.
