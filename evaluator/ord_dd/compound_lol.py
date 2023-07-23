from __future__ import annotations

from google.protobuf import json_format
from ord_schema.proto import reaction_pb2
from pydantic import BaseModel

from evaluator.ord_dd.compound_list import compound_list_greedy_matcher


def flat_list_of_lists(lol: list[list]):
    flat = []
    map_lol_to_flat = dict()
    i_flat = 0
    for i, sub_list in enumerate(lol):
        for j, item in enumerate(sub_list):
            flat.append(item)
            map_lol_to_flat[(i, j)] = i_flat
            i_flat += 1
    return flat, map_lol_to_flat


class CompoundLolDiffReport(BaseModel):
    n_misplaced_groups: int = 0

    n_ref_groups: int = 0

    n_act_groups: int = 0

    class Config:
        validate_assignment = True


def diff_compound_lol(
        lol_c1: list[list[reaction_pb2.Compound]],
        lol_c2: list[list[reaction_pb2.Compound]],
) -> CompoundLolDiffReport:
    """ determine how many compound lists in `lol_c1` are misplaced in `lol_c2` using heuristics """

    report = CompoundLolDiffReport()

    lol_cd1 = [[json_format.MessageToDict(m) for m in sublist] for sublist in lol_c1]
    lol_cd2 = [[json_format.MessageToDict(m) for m in sublist] for sublist in lol_c2]

    report.n_ref_groups = len(lol_cd1)
    report.n_act_groups = len(lol_cd2)

    lol_cd1_flat, lol_to_flat_cd1 = flat_list_of_lists(lol_cd1)
    lol_cd2_flat, lol_to_flat_cd2 = flat_list_of_lists(lol_cd2)
    flat_to_lol_cd1 = {v: k for k, v in lol_to_flat_cd1.items()}
    flat_to_lol_cd2 = {v: k for k, v in lol_to_flat_cd2.items()}

    matched_i2s = compound_list_greedy_matcher(lol_cd1_flat, lol_cd2_flat)

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

    report.n_misplaced_groups = len(misplaced_groups)

    return report
