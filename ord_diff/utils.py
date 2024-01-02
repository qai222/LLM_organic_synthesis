from __future__ import annotations

import itertools
import math
from collections.abc import MutableMapping
from enum import Enum

from deepdiff import DeepDiff
from deepdiff.helper import NotPresent
from deepdiff.model import DiffLevel, PrettyOrderedSet, REPORT_KEYS


class DeepDiffKey(str, Enum):
    values_changed = 'values_changed'
    iterable_item_removed = 'iterable_item_removed',
    iterable_item_added = 'iterable_item_added'
    dictionary_item_removed = 'dictionary_item_removed'
    dictionary_item_added = 'dictionary_item_added'
    deep_distance = 'deep_distance'


def flatten(dictionary, parent_key=None):
    """
    Taken from https://stackoverflow.com/a/62186294
    Turn a nested dictionary into a flattened dictionary
    Note if there is an integer in the path tuple, one cannot tell if it is a list index or a key,
    although usually integers are not used as keys in ord messages.

    :param dictionary: The dictionary to flatten
    :param parent_key: Argument used in recursive
    :return: A flattened dictionary where keys are `path tuples` to reach leafs
    """

    items = []
    for key, value in dictionary.items():
        if parent_key:
            new_key = list(parent_key) + [key, ]
        else:
            new_key = [key, ]
        new_key = tuple(new_key)
        if isinstance(value, MutableMapping):
            if not value.items():
                items.append((new_key, None))
            else:
                items.extend(flatten(value, new_key).items())
        elif isinstance(value, list):
            if len(value):
                for k, v in enumerate(value):
                    items.extend(flatten({k: v}, new_key).items())
            else:
                items.append((new_key, None))
        else:
            items.append((new_key, value))
    return dict(items)


def flat_deepdiff_entry(t, path_list) -> dict[tuple[str | int, ...], str | int | float | None]:
    """
    the diff entry of DeepDiff has
    1. `t`: the value (can also be a dict or list) in t1 or t2 that is different
    2. `path_list`: path (keys) to that `t`

    since `t` can be non-literal (i.e. non-leaf), this function returns the map of leaf path tuple -> leaf value
    """
    path_tuple = tuple(path_list)
    if isinstance(t, dict):
        t1 = flatten(t)
        t1_from_root = {tuple(path_list + list(k)): v for k, v in t1.items()}
    elif isinstance(t, list):
        dummy_header = "DUMMY" * 3
        dummy_t = {dummy_header: t}
        t1 = flatten(dummy_t)
        t1_from_root = {tuple(path_list + list(k)[1:]): v for k, v in t1.items()}
    elif isinstance(t, NotPresent):
        t1_from_root = {path_tuple: None}
    else:
        t1_from_root = {path_tuple: t}
    return t1_from_root


def get_dict_depth(d):
    """ get the max depth of a nested dict """
    if not isinstance(d, dict) or not d:
        return 0
    else:
        return max(get_dict_depth(v) for k, v in d.items()) + 1


def find_best_match(indices1: list[int], indices2: list[int | None], distance_matrix):
    """ given a distance matrix and two lists of indices, find the best index match """
    match_space = itertools.permutations(indices2, r=len(indices1))
    best_match_distance = math.inf
    best_match_solution = None

    for match in match_space:
        match_distance = 0
        for i1, i2 in zip(indices1, match):
            if i2 is None:
                continue
            match_distance += distance_matrix[i1][i2]
        if match_distance < best_match_distance:
            best_match_distance = match_distance
            best_match_solution = match

    assert best_match_solution is not None
    return dict(zip(indices1, [*best_match_solution]))


def parse_deepdiff(dd: DeepDiff):
    """
    given a deepdiff (tree view), return leafs that are added/removed/altered
    IMPORTANT: because we use ignore_order in deepdiff, for a change determined by deepdiff,
    the m1_path_list may be different from m2_path_list
    """
    deep_distance = 0
    paths_added = []
    paths_removed = []
    paths_altered_1 = []
    paths_altered_2 = []
    leaf_paths_added = []
    leaf_paths_removed = []
    leaf_paths_altered_1 = []
    leaf_paths_altered_2 = []
    for dd_report_key, v in dd.to_dict().items():
        dd_report_key: str
        v: PrettyOrderedSet[DiffLevel] | float
        if dd_report_key == DeepDiffKey.deep_distance.value:
            deep_distance = v
            continue
        assert dd_report_key in REPORT_KEYS  # this contains all keys from DeepDiff
        for value_altered_level in v:
            is_t1_none = isinstance(value_altered_level.t1, NotPresent)
            is_t2_none = isinstance(value_altered_level.t2, NotPresent)

            path_list_to_t1 = value_altered_level.path(output_format='list', use_t2=False)
            path_list_to_t2 = value_altered_level.path(output_format='list', use_t2=True)

            t1_leafs_from_root = flat_deepdiff_entry(value_altered_level.t1, path_list_to_t1)
            t2_leafs_from_root = flat_deepdiff_entry(value_altered_level.t2, path_list_to_t2)

            if is_t1_none and not is_t2_none:
                paths_added.append(path_list_to_t2)
                leaf_paths_added += list(t2_leafs_from_root.keys())
            elif not is_t1_none and is_t2_none:
                paths_removed.append(path_list_to_t1)
                leaf_paths_removed += list(t1_leafs_from_root.keys())
            elif not is_t1_none and not is_t2_none:
                # TODO note this assignment may not be the actual assignment for leafs:
                #  ex. I can have a sub-field in t1 removed
                paths_altered_1.append(path_list_to_t1)
                paths_altered_2.append(path_list_to_t2)
                leaf_paths_altered_1 += list(t1_leafs_from_root.keys())
                leaf_paths_altered_2 += list(t2_leafs_from_root.keys())
            else:
                raise ValueError
    return (
        deep_distance,
        paths_added, paths_removed, paths_altered_1, paths_altered_2,
        leaf_paths_added, leaf_paths_removed, leaf_paths_altered_1, leaf_paths_altered_2
    )


def flat_list_of_lists(lol: list[list]) -> tuple[list, dict[tuple[int, int], int]]:
    """
    flat to a list

    :param lol: list of lists
    :return: the flat list, a map of <tuple index of lol (i,j)> -> <flat list index>
    """
    flat = []
    map_lol_to_flat = dict()
    i_flat = 0
    for i, sub_list in enumerate(lol):
        for j, item in enumerate(sub_list):
            flat.append(item)
            map_lol_to_flat[(i, j)] = i_flat
            i_flat += 1
    return flat, map_lol_to_flat
