"""
Test suite for YAML schema examples.

This test module validates that all the YAML examples work correctly
and can be used as both documentation and test cases.
"""

from pathlib import Path

import nbclient
import nbformat
import pytest


def get_example_notebooks():
    """Find all example notebooks in the examples directory."""
    examples_dir = Path(__file__).parent.parent / "examples"
    notebooks = sorted(examples_dir.glob("*/example.ipynb"))
    return dict((nb.parent.name, nb) for nb in notebooks)


@pytest.mark.slow
@pytest.mark.parametrize(
    "notebook_path",
    get_example_notebooks().values(),
    ids=list(get_example_notebooks().keys()),
)
def test_example_notebook_runs(notebook_path):
    """
    Test that an example notebook runs without raising exceptions.
    """
    # Read the notebook
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    # Execute the notebook
    client = nbclient.NotebookClient(
        nb,
        timeout=60,
        kernel_name="python3",
        resources={"metadata": {"path": notebook_path.parent}},
    )

    try:
        client.execute()
    except nbclient.exceptions.CellExecutionError as e:
        pytest.fail(
            f"Notebook {notebook_path} failed to execute.\n"
            f"Error in cell {e.cell_index}:\n{e}"
        )
    except Exception as e:
        pytest.fail(f"Notebook {notebook_path} failed with unexpected error: {e}")
