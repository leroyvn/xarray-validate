import numpy as np
import pytest
import xarray as xr


@pytest.fixture(autouse=True, scope="session")
def add_imports_to_xdoctest_namespace(xdoctest_namespace):
    """
    Add common imports to the doctest namespace so that they are available
    without explicit imports.
    """
    xdoctest_namespace["np"] = np


@pytest.fixture
def ds():
    ds = xr.Dataset(
        {
            "x": xr.DataArray(np.arange(4) - 2, dims="x"),
            "foo": xr.DataArray(np.ones(4, dtype="i4"), dims="x"),
            "bar": xr.DataArray(
                np.arange(8, dtype=np.float64).reshape(4, 2), dims=("x", "y")
            ),
        }
    )
    return ds
