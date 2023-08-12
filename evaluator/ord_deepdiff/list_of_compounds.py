from __future__ import annotations

import math
from collections import defaultdict
from enum import Enum
from typing import Optional

from deepdiff import DeepDiff
from deepdiff.helper import NotPresent
from deepdiff.model import DiffLevel, PrettyOrderedSet, REPORT_KEYS
from google.protobuf import json_format
from ord_schema.proto import reaction_pb2

from evaluator.ord_deepdiff.base import DiffReport, DiffReportKind, FieldChangeType
from evaluator.ord_deepdiff.utils import DeepDiffKey, flatten, get_compound_name, ORD_PATH_DELIMITER, get_path_tuple, \
    get_leaf_path_tuple_to_leaf_value

CONSIDER_OTHER_COMPOUND_FIELD_CLASS = False


class CompoundFieldClass(str, Enum):
    """ they should be disjoint so any leaf of a compound can only be one of the three classes """

    reaction_role = 'reactionRole'

    identifiers = 'identifiers'

    amount = 'amount'

    @staticmethod
    def get_field_class(path_tuple: list[str | int] | tuple[str | int, ...]) -> CompoundFieldClass | None:
        """
        given a `path_tuple` (`path_list`) of a field returned by DeepDiff, return our classification
        """
        for ck in list(CompoundFieldClass):
            if ck in path_tuple:
                return ck

    @staticmethod
    def get_field_path_tuple_to_field_class(compound_dict: dict) -> dict[
        tuple[str | int, ...], CompoundFieldClass | None]:
        flat = flatten(compound_dict, separator=ORD_PATH_DELIMITER)
        d = dict()
        for k in flat:
            path_tuple = get_path_tuple(k)
            field_class = CompoundFieldClass.get_field_class(path_tuple)
            d[path_tuple] = field_class
        return d

    @staticmethod
    def get_field_class_to_field_path_tuples(compound_dict: dict) -> dict[
        CompoundFieldClass | None, list[tuple[str | int, ...]]]:
        path_to_class = CompoundFieldClass.get_field_path_tuple_to_field_class(compound_dict)
        d = defaultdict(list)
        for path, field_class in path_to_class.items():
            d[field_class].append(path)
        return d


compound_field_change_stats_type = dict[Optional[CompoundFieldClass], dict[FieldChangeType, int]]


class DiffReportListOfCompounds(DiffReport):
    kind: DiffReportKind = DiffReportKind.List_Of_COMPOUNDS

    # num of compound pairs that have at least one field added/removed/changed
    # note # of compound pairs always == n_ref_compounds
    n_altered_compounds: int = 0

    # num of fields in ref, `None` means the field class is not included in `CompoundFieldClass`
    field_counter_ref: dict[CompoundFieldClass | None, int] = dict()

    # num of fields in act, `None` means the field class is not included in `CompoundFieldClass`
    field_counter_act: dict[CompoundFieldClass | None, int] = dict()

    # how the values of these fields change
    field_change_stats: compound_field_change_stats_type = dict()

    # deep distances of compound pairs, https://zepworks.com/deepdiff/current/deep_distance.html
    # note their average may be different from direct comparison between two list with ignore_order=True
    deep_distances: list[float] = []

    # self.index_match[i] = <the matched index of Compound in self.actual> | None if no match
    index_match: dict[int, int | None] = {}

    @property
    def n_ref_compounds(self):
        """ number of compounds in the reference list """
        return len(self.reference)

    @property
    def n_act_compounds(self):
        """ number of compounds in the actual list """
        return len(self.actual)

    @property
    def n_absent_compounds(self):
        """ compounds that are absent in the act list but present in the ref list """
        return self.n_ref_compounds - self.n_act_compounds

    @property
    def n_excess_compounds(self):
        """ compounds in the act list that has no match to a ref compound """
        return self.n_act_compounds - self.n_ref_compounds

    @staticmethod
    def get_empty_field_change_stats() -> compound_field_change_stats_type:
        field_stats = dict()
        for ck in list(CompoundFieldClass) + [None, ]:
            field_stats[ck] = dict()
            for fct in list(FieldChangeType):
                field_stats[ck][fct] = 0
        return field_stats

    class Config:
        validate_assignment = True


def list_of_compounds_greedy_matcher(cds1: list[dict], cds2: list[dict]) -> list[int | None]:
    """
    for each compound in cds1, find the most similar one in cds2 based on their names, use the full dicts to break tie

    - note1: item matching for ignoring order in deepdiff is subject to parameterized deep_distance, see
    https://zepworks.com/deepdiff/current/ignore_order.html#cutoff-distance-for-pairs
    https://zepworks.com/deepdiff/current/ignore_order.html#cutoff-intersection-for-pairs

    - note2: `iterable_compare_func` could be a nice way to match compounds, unfortunately there is a bug
    see https://github.com/seperman/deepdiff/issues/307

    - note3: I have to write my own matcher as NAME carries much more weight than other fields

    :param cds1: ref list of compound dictionaries
    :param cds2: act list of compound dictionaries
    :return: matched indices of cds2, None if no match
    """
    matched_i2s = []
    for i1, cd1 in enumerate(cds1):
        name1 = get_compound_name(cd1)
        matched_i2 = None
        matched_i2_distance_name = math.inf
        matched_i2_distance_full = math.inf
        for i2, cd2 in enumerate(cds2):
            if i2 in matched_i2s:
                continue
            name2 = get_compound_name(cd2)
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


def inspect_compound_pair(
        ref_compound_dict: dict,
        act_compound_dict: dict,
) -> tuple[list[float], compound_field_change_stats_type]:
    ref_field_path_tuple_to_class = CompoundFieldClass.get_field_path_tuple_to_field_class(ref_compound_dict)
    act_field_path_tuple_to_class = CompoundFieldClass.get_field_path_tuple_to_field_class(act_compound_dict)

    field_stats = DiffReportListOfCompounds.get_empty_field_change_stats()

    dd = DeepDiff(
        ref_compound_dict, act_compound_dict,
        ignore_order=True, verbose_level=2, view='tree', get_deep_distance=True
    )

    deep_distance = 0
    for k, v in dd.to_dict().items():
        k: str
        v: PrettyOrderedSet[DiffLevel] | float

        if k == DeepDiffKey.deep_distance.value:
            deep_distance = v
        else:
            assert k in REPORT_KEYS  # this contains all keys from DeepDiff
            for value_altered_level in v:
                path_list = value_altered_level.path(output_format='list')
                t1 = value_altered_level.t1
                t2 = value_altered_level.t2
                is_ref_none = isinstance(t1, NotPresent)
                is_act_none = isinstance(t2, NotPresent)
                if is_ref_none and not is_act_none:
                    fct = FieldChangeType.ADDITION
                    t_from_root = get_leaf_path_tuple_to_leaf_value(t2, path_list)
                    field_path_tuple_to_field_class = act_field_path_tuple_to_class
                elif not is_ref_none and is_act_none:
                    fct = FieldChangeType.REMOVAL
                    t_from_root = get_leaf_path_tuple_to_leaf_value(t1, path_list)
                    field_path_tuple_to_field_class = ref_field_path_tuple_to_class
                elif not is_ref_none and not is_act_none:
                    fct = FieldChangeType.ALTERATION
                    t_from_root = get_leaf_path_tuple_to_leaf_value(t1, path_list)
                    field_path_tuple_to_field_class = ref_field_path_tuple_to_class
                else:
                    raise ValueError

                for t_key in t_from_root:
                    t_key_class = field_path_tuple_to_field_class[t_key]
                    field_stats[t_key_class][fct] += 1

    return deep_distance, field_stats


def diff_list_of_compounds(
        ref_compounds: list[reaction_pb2.Compound] | list[reaction_pb2.ProductCompound],
        act_compounds: list[reaction_pb2.Compound] | list[reaction_pb2.ProductCompound],
) -> DiffReportListOfCompounds:
    """
    find the differences between two lists of compound messages
    1. each compound in ref_compounds is greedily matched with one from act_compounds, matched with None if missing
    2. use deepdiff to inspect matched pairs

    a note about product compound measurement: yield value is often missing in the given procedure (it is calculated
    in ORD) so extracting these values is not possible for these cases.
    (tho it would be wild if this can be used for yield prediction...)
    """
    report = DiffReportListOfCompounds()

    # # are we comparing product compounds?
    # comparing_product_compound = False
    # if ref_compounds[0].__class__ == reaction_pb2.ProductCompound:
    #     comparing_product_compound = True

    ref_compounds_dicts = [json_format.MessageToDict(c) for c in ref_compounds]
    act_compounds_dicts = [json_format.MessageToDict(c) for c in act_compounds]

    report.reference = ref_compounds_dicts
    report.actual = act_compounds_dicts

    # make sure only names are included, this should already be guaranteed by data preparation scripts for LLM training
    # this is important for LLM as usually only names are included in procedures
    assert all(len(d['identifiers']) == 1 and d['identifiers'][0]['type'] == 'NAME' for d in ref_compounds_dicts)
    assert all(len(d['identifiers']) == 1 and d['identifiers'][0]['type'] == 'NAME' for d in act_compounds_dicts)

    matched_i2s = list_of_compounds_greedy_matcher(ref_compounds_dicts, act_compounds_dicts)

    # # not used for now
    # # in act
    # excess_compounds = [act_compounds_dicts[icd] for icd in range(len(act_compounds_dicts)) if icd not in matched_i2s]
    # # in ref
    # absent_compounds = [ref_compounds_dicts[icd] for icd in range(len(ref_compounds_dicts)) if matched_i2s[icd] is None]

    compound_index_pairs = dict([(i, matched_i2s[i]) for i in range(len(ref_compounds_dicts))])
    report.index_match = compound_index_pairs

    field_stats_total = DiffReportListOfCompounds.get_empty_field_change_stats()

    compound_pair_deep_distances = []
    for i, j in compound_index_pairs.items():
        ref_compound_dict = ref_compounds_dicts[i]
        if j is None:
            # TODO we may set act_compound_dict to be an empty dict...
            continue

        act_compound_dict = act_compounds_dicts[j]
        deep_distance, field_stats_pair = inspect_compound_pair(ref_compound_dict, act_compound_dict)
        compound_pair_deep_distances.append(deep_distance)

        compound_altered = False
        for ck in field_stats_total:
            for fct in field_stats_total[ck]:
                field_stats_total[ck][fct] += field_stats_pair[ck][fct]
                if field_stats_pair[ck][fct] > 0:
                    compound_altered = True
        if compound_altered:
            report.n_altered_compounds += 1

    report.deep_distances = compound_pair_deep_distances
    report.field_change_stats = field_stats_total
    return report
