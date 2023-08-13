from __future__ import annotations

from google.protobuf import json_format
from ord_schema.proto import reaction_pb2

from evaluator.ord_deepdiff.base import DiffReport, DiffReportKind
from evaluator.ord_deepdiff.list_of_compounds import list_of_compounds_exhaustive_matcher


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


class DiffReportCompoundLol(DiffReport):
    kind: DiffReportKind = DiffReportKind.LIST_OF_COMPOUND_LISTS

    n_misplaced_groups: int = 0

    @property
    def n_ref_groups(self):
        return len(self.reference)

    @property
    def n_act_groups(self):
        return len(self.actual)


def diff_list_of_compound_lists(
        lol_c1: list[list[reaction_pb2.Compound]],
        lol_c2: list[list[reaction_pb2.Compound]],
) -> DiffReportCompoundLol:
    """ determine how many compound lists in `lol_c1` are misplaced in `lol_c2` using heuristics """

    report = DiffReportCompoundLol()

    lol_cd1 = [[json_format.MessageToDict(m) for m in sublist] for sublist in lol_c1]
    lol_cd2 = [[json_format.MessageToDict(m) for m in sublist] for sublist in lol_c2]

    report.reference = lol_cd1
    report.actual = lol_cd2

    lol_cd1_flat, lol_to_flat_cd1 = flat_list_of_lists(lol_cd1)
    lol_cd2_flat, lol_to_flat_cd2 = flat_list_of_lists(lol_cd2)
    flat_to_lol_cd1 = {v: k for k, v in lol_to_flat_cd1.items()}
    flat_to_lol_cd2 = {v: k for k, v in lol_to_flat_cd2.items()}

    # matched_i2s = list_of_compounds_greedy_matcher(lol_cd1_flat, lol_cd2_flat)
    matched_i2s = list_of_compounds_exhaustive_matcher(lol_cd1_flat, lol_cd2_flat)

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
