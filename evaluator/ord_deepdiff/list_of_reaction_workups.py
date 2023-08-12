from __future__ import annotations

import itertools
import math
from collections import defaultdict

from deepdiff import DeepDiff
from google.protobuf import json_format
from ord_schema.proto import reaction_pb2

from evaluator.ord_deepdiff.base import DiffReport, DiffReportKind

# TODO use `WorkupFieldClass` to describe changes at leaf levels

WORKUP_TYPES = list(reaction_pb2.ReactionWorkup.ReactionWorkupType.keys())


class DiffReportListOfReactionWorkups(DiffReport):
    kind: DiffReportKind = DiffReportKind.LIST_OF_REACTION_WORKUPS

    n_workups_absent: int = 0

    n_workups_excess: int = 0

    index_match: dict[int, int | None] = dict()

    @property
    def n_workups_ref(self):
        return len(self.reference)

    @property
    def n_workups_act(self):
        return len(self.actual)


def list_of_reaction_workups_exhaustive_matcher(rws1: list[dict], rws2: list[dict]) -> list[int | None]:
    assert len(rws1) > 0

    indices1 = []
    plausible_indices2 = []
    for i1, rw1 in enumerate(rws1):
        rw1_type = rw1['type']
        for i2, rw2 in enumerate(rws2):
            rw2_type = rw2['type']
            if rw2_type == rw1_type:
                plausible_indices2.append(i2)
        indices1.append(i1)
    indices2 = sorted(set(plausible_indices2))

    # get distance matrix
    dist_mat = defaultdict(dict)
    for i1, rw1 in enumerate(rws1):
        rw1_type = rw1['type']
        for i2 in indices2:
            rw2 = rws2[i2]
            rw2_type = rw2['type']
            try:
                distance_type = DeepDiff(rw1_type, rw2_type, get_deep_distance=True).to_dict()['deep_distance']
            except KeyError:
                distance_type = 0
            try:
                distance_full = DeepDiff(rw1, rw2, get_deep_distance=True, ignore_order=True).to_dict()['deep_distance']
            except KeyError:
                distance_full = 0
            distance = distance_type * 100 + distance_full  # large penalty for wrong names
            dist_mat[i1][i2] = distance

    while len(indices2) < len(indices1):
        indices2.append(None)

    match_space = itertools.permutations(indices2, r=len(indices1))
    best_match_distance = math.inf
    best_match_solution = None

    for match in match_space:
        match_distance = 0
        for i1, i2 in zip(indices1, match):
            if i2 is None:
                continue
            match_distance += dist_mat[i1][i2]
        if match_distance < best_match_distance:
            best_match_distance = match_distance
            best_match_solution = match

    assert best_match_solution is not None
    for i1, i2 in zip(indices1, best_match_solution):
        if i2 is None:
            continue
    return best_match_solution


def diff_list_of_reaction_workups(
        ref_reaction_workups: list[reaction_pb2.ReactionWorkup] | list[reaction_pb2.ReactionWorkup],
        act_reaction_workups: list[reaction_pb2.ReactionWorkup] | list[reaction_pb2.ReactionWorkup],
) -> DiffReportListOfReactionWorkups:
    report = DiffReportListOfReactionWorkups()

    ref_reaction_workups_dicts = [json_format.MessageToDict(c) for c in ref_reaction_workups]
    act_reaction_workups_dicts = [json_format.MessageToDict(c) for c in act_reaction_workups]

    report.reference = ref_reaction_workups_dicts
    report.actual = act_reaction_workups_dicts

    matched_i2s = list_of_reaction_workups_exhaustive_matcher(ref_reaction_workups_dicts, act_reaction_workups_dicts)
    compound_index_pairs = dict([(i, matched_i2s[i]) for i in range(len(ref_reaction_workups_dicts))])
    report.index_match = compound_index_pairs

    # in act
    report.n_workups_excess = len([i2 for i2 in range(len(act_reaction_workups_dicts)) if i2 not in matched_i2s])

    # in ref
    report.n_workups_absent = len([i1 for i1, i2 in compound_index_pairs if i2 is None])
    return report
