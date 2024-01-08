from __future__ import annotations

import pandas as pd

from ord_diff.base import CompoundLeafType, DeltaType
from ord_diff.schema import MDict, MDictListDiff, MDictDiff, MessageType, get_compound_leaf_type
from ord_diff.utils import flat_list_of_lists


def get_compound_leaf_type_counter(cd: MDict):
    counter = {clt: 0 for clt in list(CompoundLeafType) + [None, ]}
    for leaf in cd.leafs:
        counter[get_compound_leaf_type(leaf)] += 1
    return counter


def report_diff(
        diff: MDictDiff, message_type: MessageType = None
):
    records = []
    for leaf in diff.md1.leafs:
        if leaf in diff.delta_leafs[DeltaType.REMOVAL]:
            ct = DeltaType.REMOVAL
        elif leaf in diff.delta_leafs[DeltaType.ALTERATION]:
            ct = DeltaType.ALTERATION
        else:
            ct = None
        record = {
            "from": "m1",
            "path": ".".join([str(p) for p in leaf.path_list]),
            "change_type": ct,
            "is_explicit": leaf.is_explicit,
            "value": leaf.value,
        }
        if message_type in [MessageType.COMPOUND, MessageType.PRODUCT_COMPOUND]:
            record['leaf_type'] = get_compound_leaf_type(leaf)
        records.append(record)
    for leaf in diff.md2.leafs:
        if leaf in diff.delta_leafs[DeltaType.ADDITION]:
            record = {
                "from": "m2",
                "path": ".".join([str(p) for p in leaf.path_list]),
                "change_type": DeltaType.ADDITION,
                "is_explicit": leaf.is_explicit,
                "value": leaf.value,
            }
            if message_type == MessageType.COMPOUND:
                record['leaf_type'] = get_compound_leaf_type(leaf)
            records.append(record)
    return pd.DataFrame.from_records(records)


def report_diff_list(
        compound_list_diff: MDictListDiff,
        message_type: MessageType,
):
    dfs = []
    for i, diff in enumerate(compound_list_diff.pair_comparisons):
        if diff is None:
            continue
        df = report_diff(diff, message_type=message_type)
        df['pair_index'] = i
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def get_misplaced_compound_lol(
        lol_cd1: list[list[MDict]],
        lol_cd2: list[list[MDict]],
):
    """ determine how many compound lists in `lol_c1` are misplaced in `lol_c2` """
    lol_cd1_flat, lol_to_flat_cd1 = flat_list_of_lists(lol_cd1)
    lol_cd2_flat, lol_to_flat_cd2 = flat_list_of_lists(lol_cd2)
    flat_to_lol_cd1 = {v: k for k, v in lol_to_flat_cd1.items()}
    flat_to_lol_cd2 = {v: k for k, v in lol_to_flat_cd2.items()}

    matched_i2s = MDictListDiff.get_index_match(lol_cd1_flat, lol_cd2_flat, message_type=MessageType.COMPOUND)
    lol1_to_lol2 = dict()
    for lol_index_1 in lol_to_flat_cd1:
        flat_index_1 = lol_to_flat_cd1[lol_index_1]
        matched_index_2 = matched_i2s[flat_index_1]
        if matched_index_2 is None:
            lol_index_2 = None
        else:
            lol_index_2 = flat_to_lol_cd2[matched_index_2]
        lol1_to_lol2[lol_index_1] = lol_index_2

    misplaced_groups = []
    for i, group in enumerate(lol_cd1):
        is_misplaced = False
        group_indices_1 = [(i, j) for j in range(len(group))]
        group_indices_2 = [lol1_to_lol2[gi] for gi in group_indices_1]
        if None in group_indices_2:
            # print("group element missing")
            is_misplaced = True
        elif len(set([x[0] for x in group_indices_2])) != 1:
            # print("group split")
            is_misplaced = True
        elif len(lol_cd2[group_indices_2[0][0]]) > len(group):
            # print("group expanded")
            is_misplaced = True
        elif len(lol_cd2[group_indices_2[0][0]]) < len(group):
            # print("group contracted")  # never happens as the matching algo fills None
            is_misplaced = True
        if is_misplaced:
            misplaced_groups.append(group)
    return misplaced_groups
