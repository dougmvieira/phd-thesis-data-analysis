from typing import (
    Any, Callable, Iterable, Mapping, Sequence, Tuple, TypeVar, Union
)

import numpy as np
from numpy.typing import DTypeLike, NDArray

T = TypeVar('T')
S = TypeVar('S')


def assert_arrays_are_1d(*arrs: NDArray[T]) -> None:
    for arr in arrs:
        if arr.ndim != 1:
            raise NotImplementedError("Arrays must be 1-dim")


def len_unique(arrs: Iterable[NDArray[T]]) -> int:
    # Specialised for `NDArray[T]` for clearer error messages
    len_set = set(map(len, arrs))
    assert len_set, "No arrays provided"
    uniq = len_set.pop()
    assert not len_set, "Array lengths mismatch"
    return uniq


def is_sorted(arr: NDArray[T]) -> bool:
    try:
        return np.all(arr[:-1] < arr[1:])
    except TypeError:
        # TODO: This branch is for structured arrays, needs work.
        return np.all(np.sort(arr) == arr)


def as_struct_array(arr: NDArray[T], name: str) -> NDArray[T]:
    return arr.astype([(name, arr.dtype)])


def structured_from_arrays(**arrs: NDArray[T]) -> NDArray[S]:
    assert_arrays_are_1d(*arrs.values())
    n_rows = len_unique(arrs.values())
    dtype = [(name, arr.dtype) for name, arr in arrs.items()]
    structured = np.empty(n_rows, dtype=dtype)
    for name, arr in arrs.items():
        structured[name] = arr
    return structured


def groupby_1d(arr: NDArray[T]) -> Tuple[NDArray[T], NDArray[np.int_]]:
    return np.unique(arr, return_inverse=True)


def groupby_nd(**arrs: NDArray[Any]) -> Tuple[NDArray[T], NDArray[np.int_]]:
    return groupby_1d(structured_from_arrays(**arrs))


def groupby(
    *arrs: NDArray[T], **kwarrs: NDArray[Any]
) -> Tuple[NDArray[T], NDArray[np.int_]]:
    assert not (arrs and kwarrs), (
        "Cannot groupby labeled and unlabeled arrays simultaneously"
    )
    if arrs and len(arrs) == 1:
        return groupby_1d(arrs[0])
    elif arrs:
        raise NotImplementedError("Cannot groupby multiple unlabeled arrays")
    elif kwarrs:
        return groupby_nd(**kwarrs)
    else:
        raise AssertionError("No arrays provided")


def aggregate_inplace(
    fun: Callable[[NDArray[T]], S],
    arr: NDArray[T],
    group_keys: NDArray[T],
    group_pos: NDArray[np.int_],
    group_vals: NDArray[S],
) -> None:
    assert is_sorted(group_keys), "`group_keys` is not sorted"
    for group_id in range(len(group_keys)):
        group_vals[group_id] = fun(arr[group_pos == group_id])


def transform_inplace(
    fun: Callable[[NDArray[T]], NDArray[S]],
    arr: NDArray[T],
    group_keys: NDArray[T],
    group_pos: NDArray[np.int_],
    group_vals: NDArray[S],
) -> None:
    assert is_sorted(group_keys), "`group_keys` is not sorted"
    for group_id in range(len(group_keys)):
        group_mask = group_pos == group_id
        group_vals[group_mask] = fun(arr[group_mask])


def aggregate(
    fun: Callable[[NDArray[T]], S],
    arr: NDArray[T],
    group_keys: NDArray[T],
    group_pos: NDArray[np.int_],
    group_vals_dtype: DTypeLike,
) -> NDArray[S]:
    group_vals = np.empty(len(group_keys), dtype=group_vals_dtype)
    aggregate_inplace(fun, arr, group_keys, group_pos, group_vals)
    return group_vals


def transform(
    fun: Callable[[NDArray[T]], NDArray[S]],
    arr: NDArray[T],
    group_keys: NDArray[T],
    group_pos: NDArray[np.int_],
    group_vals_dtype: DTypeLike,
) -> NDArray[S]:
    group_vals = np.empty(len(arr), dtype=group_vals_dtype)
    transform_inplace(fun, arr, group_keys, group_pos, group_vals)
    return group_vals


def test_aggregate():
    data = np.array(
        [
            ('F', 10, 1.3),
            ('F', 10, 1.4),
            ('F', 15, 1.6),
            ('F', 20, 1.7),
            ('M', 10, 1.4),
            ('M', 20, 1.75),
            ('M', 20, 1.7),
            ('M', 20, 1.8),
        ],
        dtype=[('sex', 'U1'), ('age', 'i8'), ('height', 'f8')]
    )
    averages = np.array(
        [
            ('F', 10, 1.35),
            ('F', 15, 1.6),
            ('F', 20, 1.7),
            ('M', 10, 1.4),
            ('M', 20, 1.75),
        ],
        dtype=[('sex', 'U1'), ('age', 'i8'), ('height', 'f8')]
    )
    group_keys, group_pos = groupby(data[['sex', 'age']])
    group_vals = aggregate(
        np.mean, data['height'], group_keys, group_pos, np.float64
    )
    np.testing.assert_array_equal(averages[['sex', 'age']], group_keys)
    np.testing.assert_array_equal(averages['height'], group_vals)


test_aggregate()
