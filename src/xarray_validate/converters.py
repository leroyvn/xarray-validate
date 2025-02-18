from __future__ import annotations

import numpy as np


def array_type_converter(value):
    if value == "<class 'dask.array.core.Array'>":
        import dask.array as da

        return da.Array
    elif value == "<class 'numpy.ndarray'>":
        return np.ndarray
    else:
        return value
