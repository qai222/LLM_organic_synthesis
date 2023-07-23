from __future__ import annotations

import math

from deepdiff import DeepDiff
from deepdiff.model import DiffLevel, PrettyOrderedSet
from google.protobuf import json_format
from ord_schema.proto import reaction_pb2
from pydantic import BaseModel

from evaluator.ord_dd.utils import DeepDiffKey, DeepDiffError


def compound_list_greedy_matcher(cds1: list[dict], cds2: list[dict]) -> list[int | None]:
    """
    for each compound in cds1, find the most similar one in cds2 based on their names, use the full dicts to break tie

    note1: item matching for ignoring order in deepdiff is subject to parameterized deep_distance, see
    https://zepworks.com/deepdiff/current/ignore_order.html#cutoff-distance-for-pairs
    https://zepworks.com/deepdiff/current/ignore_order.html#cutoff-intersection-for-pairs

    note2: `iterable_compare_func` could be a nice way to match compounds, unfortunately there is a bug
    see https://github.com/seperman/deepdiff/issues/307

    note3: I have to write my own matcher as NAME carries much more weight than other fields

    :param cds1: ref list of compound dictionaries
    :param cds2: act list of compound dictionaries
    :return: matched indices of cds2, None if no match
    """
    matched_i2s = []
    for i1, cd1 in enumerate(cds1):
        name1 = cd1['identifiers'][0]['value']
        matched_i2 = None
        matched_i2_distance_name = math.inf
        matched_i2_distance_full = math.inf
        for i2, cd2 in enumerate(cds2):
            if i2 in matched_i2s:
                continue
            name2 = cd2['identifiers'][0]['value']
            try:
                distance_name = DeepDiff(name1, name2, get_deep_distance=True).to_dict()['deep_distance']
            except KeyError:
                distance_name = 0
            try:
                distance_full = DeepDiff(cd1, cd2, get_deep_distance=True, ignore_order=True).to_dict()['deep_distance']
            except KeyError:
                distance_full = 0
            if matched_i2 is None or distance_name < matched_i2_distance_name:
                matched_i2 = i2
                matched_i2_distance_name = distance_name
                matched_i2_distance_full = distance_full
            elif distance_name == matched_i2_distance_name:
                if distance_full < matched_i2_distance_full:
                    matched_i2 = i2
                    matched_i2_distance_full = distance_full
        assert len(matched_i2s) == i1
        matched_i2s.append(matched_i2)
    return matched_i2s


class CompoundListDiffReport(BaseModel):
    # number of compounds in the reference list
    n_ref_compounds: int = 0

    # number of compounds in the actual list
    n_act_compounds: int = 0

    # compounds that are absent in the act list but present in the ref list,
    # this can only happen when len(act) < len(ref)
    n_absent_compounds: int = 0

    # compounds in the act list that has no match to a ref compound,
    # this can only happen when len(act) > len(ref)
    n_excess_compounds: int = 0

    # reaction role changed in a pair of compounds
    n_compounds_reaction_role_changed: int = 0

    # identifiers changed in a pair of compounds
    n_compounds_identifiers_changed: int = 0

    # amount changed in a pair of compounds
    n_compounds_amount_changed: int = 0

    # average deep distance of compound pairs, https://zepworks.com/deepdiff/current/deep_distance.html
    # note this maybe different from direct comparison between two list with ignore_order=True
    average_deep_distance: float = 0.0

    index_match: dict[int, int | None] = {}

    class Config:
        validate_assignment = True


def diff_compound_list(
        ref_compounds: list[reaction_pb2.Compound],
        act_compounds: list[reaction_pb2.Compound],
) -> CompoundListDiffReport:
    """
    find the differences between two lists of compound messages
    1. each compound in ref_compounds is greedily matched with one from act_compounds, matched with None if missing
    2. use deepdiff to inspect matched pairs
    """
    report = CompoundListDiffReport()

    report.n_ref_compounds = len(ref_compounds)
    report.n_act_compounds = len(act_compounds)

    ref_compounds_dicts = [json_format.MessageToDict(c) for c in ref_compounds]
    act_compounds_dicts = [json_format.MessageToDict(c) for c in act_compounds]

    # make sure only names are included, this should already be guaranteed by data preparation scripts
    assert all(len(d['identifiers']) == 1 and d['identifiers'][0]['type'] == 'NAME' for d in ref_compounds_dicts)
    assert all(len(d['identifiers']) == 1 and d['identifiers'][0]['type'] == 'NAME' for d in act_compounds_dicts)

    matched_i2s = compound_list_greedy_matcher(ref_compounds_dicts, act_compounds_dicts)
    # in act
    excess_compounds = [act_compounds_dicts[icd] for icd in range(len(act_compounds_dicts)) if icd not in matched_i2s]
    # in ref
    absent_compounds = [ref_compounds_dicts[icd] for icd in range(len(ref_compounds_dicts)) if matched_i2s[icd] is None]

    report.n_excess_compounds = len(excess_compounds)
    report.n_absent_compounds = len(absent_compounds)

    compounds_reaction_role_changed = []
    compounds_identifiers_changed = []
    compounds_amount_changed = []
    compound_pair_deep_distances = []
    compound_index_pairs = dict([(i, matched_i2s[i]) for i in range(len(ref_compounds_dicts))])
    report.index_match = compound_index_pairs
    for i, j in compound_index_pairs.items():
        if j is None:
            continue
        dd = DeepDiff(
            ref_compounds_dicts[i], act_compounds_dicts[j],
            ignore_order=True, verbose_level=2, view='tree', get_deep_distance=True
        )
        deep_distance = 0
        has_reaction_role_changed = False
        has_identifiers_changed = False
        has_amount_changed = False
        for k, v in dd.to_dict().items():
            k: str
            v: PrettyOrderedSet[DiffLevel] | float
            if k == DeepDiffKey.deep_distance.value:
                deep_distance = v
            # elif k == DeepDiffKey.values_changed.value:
            else:  # this means we consider adding/removing as changes
                for value_changed_level in v:
                    path_list = value_changed_level.path(output_format='list')
                    if 'reactionRole' in path_list:
                        has_reaction_role_changed = True
                    elif 'identifiers' in path_list:
                        has_identifiers_changed = True
                    elif 'amount' in path_list:
                        has_amount_changed = True
                    else:
                        raise DeepDiffError(
                            "`path_list` should contain and only contain one of reactionRole, identifiers, and amount."
                            f"But we got: {path_list}"
                        )

        compound_pair_deep_distances.append(deep_distance)
        if has_reaction_role_changed:
            compounds_reaction_role_changed.append((i, j))
        if has_identifiers_changed:
            compounds_identifiers_changed.append((i, j))
        if has_amount_changed:
            compounds_amount_changed.append((i, j))

    report.n_compounds_reaction_role_changed = len(compounds_reaction_role_changed)
    report.n_compounds_identifiers_changed = len(compounds_identifiers_changed)
    report.n_compounds_amount_changed = len(compounds_amount_changed)
    report.average_deep_distance = sum(compound_pair_deep_distances) / len(compound_pair_deep_distances)
    return report
