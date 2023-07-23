from __future__ import annotations

import enum


class DeepDiffKey(enum.Enum):
    values_changed = 'values_changed'
    iterable_item_removed = 'iterable_item_removed',
    iterable_item_added = 'iterable_item_added'
    dictionary_item_removed = 'dictionary_item_removed'
    dictionary_item_added = 'dictionary_item_added'
    deep_distance = 'deep_distance'


class DeepDiffError(Exception):
    pass
