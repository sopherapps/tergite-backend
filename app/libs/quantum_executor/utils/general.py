# This code is part of Tergite
#
# (C) Axel Andersson (2022)
# (C) Martin Ahindura (2025)
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#
# Refactored by Martin Ahindura (2024)

import json
import sys
from typing import List, TypeVar, Iterator, Union, Generator, Iterable

from app.libs.storage_file.file import JOB_HDF5_FILE_DELIMITER

_T = TypeVar("_T")


def find(haystack, needle):
    if not isinstance(haystack, dict):
        return

    if needle in haystack:
        yield (needle,)

    for key, val in haystack.items():
        for sub_path in find(val, needle):
            yield key, *sub_path


def freeze(d: dict) -> frozenset:
    """
    Turns a dictionary into a hashable object.
    Only works for dictionaries containing only hashable atoms.

    e.g.
    freeze({"a" : 1 , "b" : 2, "c" : { "d" : 10 }}) = frozenset({('a', 1), ('b', 2), ('c', frozenset({('d', 10)}))})
    """
    return frozenset(
        (k, v) if type(v) != dict else (k, freeze(v)) for k, v in d.items()
    )


def ceil4(n):
    """FIXME: What exactly is this doing and why?"""
    return n + (4 - n) % 4


def rot_left(arr: iter, steps: int) -> iter:
    if type(arr) != list:
        if hasattr(arr, "tolist"):
            arr = arr.tolist()
        elif hasattr(arr, "to_list"):
            arr = arr.to_list()
        else:
            arr = list(arr)

    return arr[steps:] + arr[:steps]


# taken from Python 3.10 source, because cant install 3.10 but just need this function
def insort_left(a, x, lo=0, hi=None, *, key=None):
    """Insert item x in list a, and keep it sorted assuming a is sorted.
    If x is already in a, insert it to the left of the leftmost x.
    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    if key is None:
        lo = bisect_left(a, x, lo, hi)
    else:
        lo = bisect_left(a, key(x), lo, hi, key=key)
    a.insert(lo, x)


# taken from Python 3.10 source, because cant install 3.10 but just need this function
def bisect_left(a, x, lo=0, hi=None, *, key=None):
    """Return the index where to insert item x in list a, assuming a is sorted.
    The return value i is such that all e in a[:i] have e < x, and all e in
    a[i:] have e >= x.  So if x already appears in the list, a.insert(i, x) will
    insert just before the leftmost x already there.
    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    if lo < 0:
        raise ValueError("lo must be non-negative")
    if hi is None:
        hi = len(a)
    # Note, the comparison uses "<" to match the
    # __lt__() logic in list.sort() and in heapq.
    if key is None:
        while lo < hi:
            mid = (lo + hi) // 2
            if a[mid] < x:
                lo = mid + 1
            else:
                hi = mid
    else:
        while lo < hi:
            mid = (lo + hi) // 2
            if key(a[mid]) < x:
                lo = mid + 1
            else:
                hi = mid
    return lo


# taken from Python 3.10 source, because cant install 3.10 but just need this function
def bisect_right(a, x, lo=0, hi=None, *, key=None):
    """Return the index where to insert item x in list a, assuming a is sorted.
    The return value i is such that all e in a[:i] have e <= x, and all e in
    a[i:] have e > x.  So if x already appears in the list, a.insert(i, x) will
    insert just after the rightmost x already there.
    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    if lo < 0:
        raise ValueError("lo must be non-negative")
    if hi is None:
        hi = len(a)
    # Note, the comparison uses "<" to match the
    # __lt__() logic in list.sort() and in heapq.
    if key is None:
        while lo < hi:
            mid = (lo + hi) // 2
            if x < a[mid]:
                hi = mid
            else:
                lo = mid + 1
    else:
        while lo < hi:
            mid = (lo + hi) // 2
            if x < key(a[mid]):
                hi = mid
            else:
                lo = mid + 1
    return lo


# taken from Python 3.10 source, because cant install 3.10 but just need this function
def insort_right(a, x, lo=0, hi=None, *, key=None):
    """Insert item x in list a, and keep it sorted assuming a is sorted.
    If x is already in a, insert it to the right of the rightmost x.
    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """
    if key is None:
        lo = bisect_right(a, x, lo, hi)
    else:
        lo = bisect_right(a, key(x), lo, hi, key=key)
    a.insert(lo, x)


def load_config(filepath: str):
    with open(filepath, mode="r") as _j:
        try:
            return json.load(_j)
        except json.JSONDecodeError as jsde:
            print(
                f"Could not read configuration .json file {_j.name}. Incorrectly formatted?",
                file=sys.stderr,
            )
            raise jsde


def get_duplicates(texts: List[str]) -> List[str]:
    """Gets the duplicates in a given list of strings

    Args:
        texts: the list of strings

    Returns:
        the list of duplicate strings
    """
    duplicates = []
    seen = set()
    for name in texts:
        if name in seen:
            duplicates.append(name)
        else:
            seen.add(name)
    return duplicates


def get_experiment_name(qobj_expt_name: str, expt_index: int):
    """
    Creates a cleaned version of a given experiment name

    Args:
        qobj_expt_name: the name as got from the qobject
        expt_index: the index of the experiment in the list of experiments in the qobject

    Returns:
        a sanitized name to use internally
    """
    name = "".join(x for x in qobj_expt_name if x.isalnum() or x in " -_,.()")
    return f"{name}{JOB_HDF5_FILE_DELIMITER}{expt_index}"


def flatten_list(
    items: Union[
        Iterator[Iterator[_T]],
        Generator[Iterable[_T], None, None],
        Iterable[Iterable[_T]],
    ]
) -> List[_T]:
    """Flattens a list of lists

    Args:
        items: the nested list to flatten

    Returns:
        the list flattened list
    """
    return [item for sublist in items for item in sublist]
