import numpy as np
import pytest
import xarray as xr

# ------------------------------------------------------------------------------
#                                 Pytest config
# ------------------------------------------------------------------------------


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    """Configure pytest based on command-line options."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselected by default, use --runslow to run)",
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command-line options."""
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# ------------------------------------------------------------------------------
#                                    Fixtures
# ------------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="session")
def add_imports_to_xdoctest_namespace(xdoctest_namespace):
    """
    Add common imports to the doctest namespace so that they are available
    without explicit imports.
    """
    xdoctest_namespace["np"] = np


@pytest.fixture
def ds():
    """
    Test dataset fixture.
    """
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
