from __future__ import annotations

import enum
from collections.abc import MutableMapping
from deepdiff.helper import NotPresent

ORD_PATH_DELIMITER = "..."


def get_compound_name(compound_dict: dict):
    for identifier in compound_dict["identifiers"]:
        if identifier['type'] == 'NAME':
            return identifier['value']
    raise ValueError('`NAME` not found in the compound dict')


class DeepDiffKey(enum.Enum):
    values_changed = 'values_changed'
    iterable_item_removed = 'iterable_item_removed',
    iterable_item_added = 'iterable_item_added'
    dictionary_item_removed = 'dictionary_item_removed'
    dictionary_item_added = 'dictionary_item_added'
    deep_distance = 'deep_distance'


class DeepDiffError(Exception):
    pass


def flatten(dictionary, parent_key=False, separator='.'):
    """
    taken from https://stackoverflow.com/a/62186294
    Turn a nested dictionary into a flattened dictionary

    :param dictionary: The dictionary to flatten
    :param parent_key: The string to prepend to dictionary's keys
    :param separator: The string used to separate flattened keys
    :return: A flattened dictionary
    """

    items = []
    for key, value in dictionary.items():
        new_key = str(parent_key) + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            if not value.items():
                items.append((new_key, None))
            else:
                items.extend(flatten(value, new_key, separator).items())
        elif isinstance(value, list):
            if len(value):
                for k, v in enumerate(value):
                    items.extend(flatten({str(k): v}, new_key, separator).items())
            else:
                items.append((new_key, None))
        else:
            items.append((new_key, value))
    return dict(items)


def get_path_tuple(path_str, delimiter=ORD_PATH_DELIMITER, restore_int=True) -> tuple[str | int, ...]:
    """
    given a key of a flattened nested dict, return its `path_list`

    :param path_str:
    :param delimiter: the delimiter used in the `path_str`
    :param restore_int: `utils.flatten` stringify int keys, this option restores their types to int
    :return:
    """
    path_list = path_str.split(delimiter)
    if restore_int:
        leaf = path_list[-1]
        path_list_int = []
        for k in path_list[:-1]:
            try:
                path_list_int.append(int(k))
            except ValueError:
                path_list_int.append(k)
        path_list_int.append(leaf)
        return tuple(path_list_int)
    else:
        return tuple(path_list)


def get_leaf_path_tuple_to_leaf_value(t, path_list, delimiter=ORD_PATH_DELIMITER):
    """
    the diff entry of DeepDiff has
    1. `t`: the value (can be a dict or list) in t1 or t2 that is different
    2. `path_list`: path (keys) to that value

    since `t` can be non-literal (i.e. non-leaf), this function returns the map of leaf path tuple -> leaf value
    """
    path_tuple = tuple(path_list)
    if isinstance(t, (list, dict)):
        t1 = flatten(t, separator=delimiter)
        t1_from_root = {tuple(path_list + list(get_path_tuple(k))): v for k, v in t1.items()}
    elif isinstance(t, NotPresent):
        t1_from_root = {path_tuple: None}
    else:
        t1_from_root = {path_tuple: t}
    return t1_from_root
