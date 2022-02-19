from hashlib import sha1

import numpy as np
import pandas as pd
import xarray as xr


A4_HEIGHT = 11.7
A4_WIDTH = 8.27


def is_consecutive_unique(array):
    """
    Note: In a sequence of identical elements, the last is marked unique.
    """
    is_value_unique = np.roll(array, -1) != array
    is_value_unique[-1] = True
    if array.dtype == np.float64:
        isnan_array = np.isnan(array)
        is_nan_unique = ~(np.roll(isnan_array, -1) & isnan_array)
        is_nan_unique[-1] = True
        is_unique = is_value_unique & is_nan_unique
    else:
        is_unique = is_value_unique
    return is_unique if is_unique.ndim == 1 else np.any(is_unique, 1)


def get_col(df, name):
    return (df[name].values if name in df.columns
            else df.index.get_level_values(name).values)


def cols_to_kwargs(df, **kwargs_map):
    return {kw: get_col(df, col) for kw, col in kwargs_map.items()}


def cols_to_args(df, *cols):
    return (get_col(df, col) for col in cols)


def resample(df, index):
    empty = pd.DataFrame(np.nan, pd.Index(index, name=df.index.name),
                         df.columns)
    resampled = pd.concat([df, empty])
    resampled.sort_index(inplace=True)
    resampled.ffill(inplace=True)
    resampled = resampled.loc[index]
    return resampled.loc[is_consecutive_unique(resampled.index)]


def put_call_parity(call=None, put=None, underlying=None, strike=None,
                    discount=None):
    if call is None:
        return put + underlying - strike*discount
    if put is None:
        return call - underlying + strike*discount
    if underlying is None:
        return call - put + strike*discount
    if strike is None:
        return (underlying - call + put)/discount
    if discount is None:
        return (underlying - call + put)/strike


def read_source(filename):
    with open(filename) as f:
        source = f.read()
    return source


def hash_dataset(dataset):
    h = sha1()
    for k, v in dataset.reset_coords().items():
        h.update(k.encode('utf-8'))
        h.update(v.data.tobytes())
    return dataset.assign_attrs({'lazymaker_hash': h.hexdigest()})


def combine_dataarrays(f, dim, dataarray, other, map=map, **kwargs):
    xs = np.unique(dataarray[dim])
    da_slice_iter = (dataarray.sel({dim: x}) for x in xs)
    ot_slice_iter = (other.sel({dim: x}) for x in xs)
    res_slice_iter = map(f, da_slice_iter, ot_slice_iter)
    res = xr.DataArray(dataarray.copy(), **kwargs)
    for x, res_slice in zip(xs, res_slice_iter):
        res.loc[{dim: x}] = res_slice
    return res


def equal_split(data, dim, n):
    lower = data[dim].isel({dim:  0}).data
    upper = data[dim].isel({dim: -1}).data
    bins = lower + np.arange(n + 1)*(upper - lower)/n
    return data.groupby_bins(dim, bins)
