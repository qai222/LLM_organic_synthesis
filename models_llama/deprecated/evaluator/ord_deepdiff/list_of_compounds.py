from __future__ import annotations

import itertools
import math
from collections import Counter
from collections import defaultdict
from enum import Enum
from typing import Optional

import numpy as np
from deepdiff import DeepDiff
from deepdiff.helper import NotPresent
from deepdiff.model import DiffLevel, PrettyOrderedSet, REPORT_KEYS
from google.protobuf import json_format
from loguru import logger
from ord_schema.proto import reaction_pb2

from models_llama.deprecated.evaluator.ord_deepdiff.base import DiffReport, DiffReportKind, FieldChangeType
from models_llama.deprecated.evaluator.ord_deepdiff.utils import DeepDiffKey, flatten, get_compound_name, ORD_PATH_DELIMITER, get_path_tuple, \
    get_leaf_path_tuple_to_leaf_value


class CompoundFieldClass(str, Enum):
    """ they should be "disjoint" so any leaf of a compound can only be one of the these classes """

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
        CompoundFieldClass | None, list[tuple[str | int, ...]]
    ]:
        path_to_class = CompoundFieldClass.get_field_path_tuple_to_field_class(compound_dict)
        d = defaultdict(list)
        for path, field_class in path_to_class.items():
            d[field_class].append(path)
        return d


compound_field_change_stats_type = dict[Optional[CompoundFieldClass], dict[FieldChangeType, int]]


def get_empty_field_change_stats() -> compound_field_change_stats_type:
    field_stats = dict()
    for ck in list(CompoundFieldClass) + [None, ]:
        field_stats[ck] = dict()
        for fct in list(FieldChangeType):
            field_stats[ck][fct] = 0
    return field_stats


class DiffReportListOfCompounds(DiffReport):
    kind: DiffReportKind = DiffReportKind.LIST_OF_COMPOUNDS

    # num of compound pairs that have at least one field added/removed/changed
    # note # of compound pairs always == n_ref_compounds
    n_altered_compounds: int = 0

    # num of fields in ref, `None` means the field class is not included in `CompoundFieldClass`
    field_counter_ref: dict[CompoundFieldClass | None, int] = {ck: 0 for ck in list(CompoundFieldClass) + [None, ]}

    # num of fields in act, `None` means the field class is not included in `CompoundFieldClass`
    field_counter_act: dict[CompoundFieldClass | None, int] = {ck: 0 for ck in list(CompoundFieldClass) + [None, ]}

    # how the values of these fields change
    field_change_stats: compound_field_change_stats_type = get_empty_field_change_stats()

    # deep distances of compound pairs, https://zepworks.com/deepdiff/current/deep_distance.html
    # note their average may be different from direct comparison between two list with ignore_order=True
    deep_distances: list[float] = []

    # self.index_match[i] = <the matched index of Compound in self.actual> | None if no match
    index_match: dict[int, int | None] = {}

    @property
    def compound_change_stats(self) -> dict[FieldChangeType, int]:
        d = {
            FieldChangeType.ADDITION: self.n_excess_compounds,
            FieldChangeType.REMOVAL: self.n_absent_compounds,
            FieldChangeType.ALTERATION: self.n_altered_compounds,
        }
        return d

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
        n = self.n_ref_compounds - self.n_act_compounds
        if n > 0:
            return n
        else:
            return 0

    @property
    def n_excess_compounds(self):
        """ compounds in the act list that has no match to a ref compound """
        n = self.n_act_compounds - self.n_ref_compounds
        if n > 0:
            return n
        else:
            return 0


def list_of_compounds_exhaustive_matcher(cds1: list[dict], cds2: list[dict]) -> list[int | None]:
    if len(cds1) == 0:
        return []

    indices1 = [*range(len(cds1))]
    indices2 = [*range(len(cds2))]

    # get distance matrix
    dist_mat = np.zeros((len(indices1), len(indices2)))
    for i1 in indices1:
        cd1 = cds1[i1]
        name1 = get_compound_name(cd1)
        for i2 in indices2:
            cd2 = cds2[i2]
            name2 = get_compound_name(cd2)
            try:
                distance_name = DeepDiff(name1, name2, get_deep_distance=True).to_dict()['deep_distance']
            except KeyError:
                distance_name = 0
            try:
                distance_full = DeepDiff(cd1, cd2, get_deep_distance=True, ignore_order=True).to_dict()['deep_distance']
            except KeyError:
                distance_full = 0
            distance = distance_name * 100 + distance_full  # large penalty for wrong names
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


class FieldSkipRule(str, Enum):
    no_skip = "no_skip"
    skip_reaction_role = "skip_reaction_role"
    ignore_absent_in_prompt = "ignore_absent_in_prompt"
    ignore_absent_in_prompt_with_exceptions = "ignore_absent_in_prompt_with_exceptions"


def skip_in_inspect_compound_pair(leaf_key: tuple, leaf_val, prompt: str, rule: FieldSkipRule) -> bool:
    """
    :param leaf_key:
    :param leaf_val:
    :param prompt:
    :param rule: this option allows skipping leaf fields as if they are absent in both ref and act
        "no_skip":
            don't skip any field
        "skip_reaction_role":
            if a leaf field is classified as `reaction_role`, it is skipped
        "ignore_absent_in_prompt":
            if the value of a leaf is absent in the prompt, that leaf field is skipped
        "ignore_absent_in_prompt_with_exceptions":
            except the leaf field satisfies one of the followings
                - has class of `reaction_role` (values of this field are usually not present in prompt)
                - leaf key contains "units" (values of this field can be present in a different form in prompt)
    :return:
    """
    if leaf_val is None:
        raise ValueError  # leaf key should always present in ref

    value_present_in_prompt = str(leaf_val).lower() in prompt.lower()
    field_class = CompoundFieldClass.get_field_class(leaf_key)
    if rule == FieldSkipRule.no_skip:
        return False
    elif rule == FieldSkipRule.skip_reaction_role:
        if field_class == CompoundFieldClass.reaction_role:
            return True
        return False
    elif rule == FieldSkipRule.ignore_absent_in_prompt:
        if not value_present_in_prompt:
            return True
        return False
    elif rule == FieldSkipRule.ignore_absent_in_prompt_with_exceptions:
        if not value_present_in_prompt:
            if field_class == CompoundFieldClass.reaction_role:
                return False
            if "units" in leaf_key:
                return False
            return True
        return False
    else:
        raise ValueError


def inspect_compound_pair(
        ref_compound_dict: dict,
        act_compound_dict: dict,
        skip_rule: FieldSkipRule = FieldSkipRule.ignore_absent_in_prompt_with_exceptions,
        prompt: str = "",
) -> tuple[
    list[float], compound_field_change_stats_type,
    dict[CompoundFieldClass | None, int], dict[CompoundFieldClass | None, int],
]:
    """
    inspect a pair of compounds

    :param ref_compound_dict:
    :param act_compound_dict:
    :param skip_rule: this option allows skipping leaf fields as if they are absent in both ref and act
        "no_skip":
            don't skip any field
        "skip_reaction_role":
            if a leaf field is classified as `reaction_role`, it is skipped
        "ignore_absent_in_prompt":
            if the value of a leaf is absent in the prompt, that leaf field is skipped
        "ignore_absent_in_prompt_with_exceptions":
            except the leaf field satisfies one of the followings
                - has class of `reaction_role` (values of this field are usually not present in prompt)
                - leaf key contains "units" (values of this field can be present in a different form in prompt)
        NOTE: this would not affect (upstream) compound count, ex. if a compound is missing in the prompt, you will
        still have that compound appears "removed"
    :param prompt:
    :return:
    """
    ref_field_path_tuple_to_class = CompoundFieldClass.get_field_path_tuple_to_field_class(ref_compound_dict)
    act_field_path_tuple_to_class = CompoundFieldClass.get_field_path_tuple_to_field_class(act_compound_dict)

    field_stats = get_empty_field_change_stats()
    # TODO the structure of this is still a bit confusing, maybe better to have a field-keyed dictionary,
    #  ex. stats[<field_tuple>] = {"field_class": "identifiers", "change_type": "ADDITION", ...}

    dd = DeepDiff(
        ref_compound_dict, act_compound_dict,
        ignore_order=True, verbose_level=2, view='tree', get_deep_distance=True
    )

    deep_distance = 0
    skipped_leaf_keys_ref = []
    skipped_leaf_keys_act = []
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

                t_from_root_1 = get_leaf_path_tuple_to_leaf_value(t1, path_list)
                t_from_root_2 = get_leaf_path_tuple_to_leaf_value(t2, path_list)

                if is_ref_none and not is_act_none:
                    fct = FieldChangeType.ADDITION
                    field_path_tuple_to_field_class = act_field_path_tuple_to_class
                    t_from_root = t_from_root_2
                elif not is_ref_none and is_act_none:
                    fct = FieldChangeType.REMOVAL
                    field_path_tuple_to_field_class = ref_field_path_tuple_to_class
                    t_from_root = t_from_root_1
                elif not is_ref_none and not is_act_none:
                    # not this assignment is not the final assignment:
                    # ex. I can have a sub-filed in t_from_root_1 removed
                    fct = FieldChangeType.ALTERATION
                    field_path_tuple_to_field_class = ref_field_path_tuple_to_class
                    t_from_root = t_from_root_1
                else:
                    raise ValueError

                for t_leaf_key, t_leaf_val in t_from_root.items():
                    t_leaf_key_class = field_path_tuple_to_field_class[t_leaf_key]
                    if fct == FieldChangeType.ALTERATION:
                        # actual assignment
                        if t_leaf_key not in t_from_root_1 and t_leaf_key in t_from_root_2:
                            fct_leaf = FieldChangeType.ADDITION
                        elif t_leaf_key in t_from_root_1 and t_leaf_key not in t_from_root_2:
                            fct_leaf = FieldChangeType.REMOVAL
                        elif t_leaf_key in t_from_root_1 and t_leaf_key in t_from_root_2:
                            fct_leaf = FieldChangeType.ALTERATION
                        else:
                            raise ValueError
                    else:
                        fct_leaf = fct

                    logger_msg = f"{fct_leaf} at {'.'.join(str(kk) for kk in t_leaf_key)} ({t_leaf_key_class}):\n"
                    if fct_leaf == FieldChangeType.ADDITION:
                        logger_msg += f"{None} -> {t_leaf_val}"
                    elif fct_leaf == FieldChangeType.REMOVAL:
                        logger_msg += f"{t_leaf_val} -> {None}"
                    elif fct_leaf == FieldChangeType.ALTERATION:
                        logger_msg += f"{t_leaf_val} -> {t_from_root_2[t_leaf_key]}"
                    else:
                        raise ValueError

                    logger.info(logger_msg)

                    # we can only skip when ref key is available, i.e. t_from_root = t_from_root_1
                    can_skip = fct_leaf in (FieldChangeType.REMOVAL, FieldChangeType.ALTERATION,)

                    if skip_in_inspect_compound_pair(t_leaf_key, t_leaf_val, prompt, skip_rule) and can_skip:
                        logger.warning(f"this diff is skipped by the rule: {skip_rule}")
                        if fct_leaf in (FieldChangeType.REMOVAL, FieldChangeType.ALTERATION,):
                            skipped_leaf_keys_ref.append(t_leaf_key)
                            if fct_leaf == FieldChangeType.ALTERATION:
                                skipped_leaf_keys_act.append(t_leaf_key)
                    else:
                        field_stats[t_leaf_key_class][fct_leaf] += 1

    ref_field_path_tuple_to_class_exclude_skipped = {k: v for k, v in ref_field_path_tuple_to_class.items() if
                                                     k not in skipped_leaf_keys_ref}
    act_field_path_tuple_to_class_exclude_skipped = {k: v for k, v in act_field_path_tuple_to_class.items() if
                                                     k not in skipped_leaf_keys_act}

    field_counter_ref = Counter(ref_field_path_tuple_to_class_exclude_skipped.values())
    field_counter_act = Counter(act_field_path_tuple_to_class_exclude_skipped.values())
    return deep_distance, field_stats, field_counter_ref, field_counter_act


def diff_list_of_compounds(
        ref_compounds: list[reaction_pb2.Compound] | list[reaction_pb2.ProductCompound],
        act_compounds: list[reaction_pb2.Compound] | list[reaction_pb2.ProductCompound],
        prompt: str = "",
        skip_rule: FieldSkipRule = FieldSkipRule.ignore_absent_in_prompt_with_exceptions,
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

    # matched_i2s = list_of_compounds_greedy_matcher(ref_compounds_dicts, act_compounds_dicts)
    matched_i2s = list_of_compounds_exhaustive_matcher(ref_compounds_dicts, act_compounds_dicts)

    # # not used for now
    # # in act
    # excess_compounds = [act_compounds_dicts[icd] for icd in range(len(act_compounds_dicts)) if icd not in matched_i2s]
    # # in ref
    # absent_compounds = [ref_compounds_dicts[icd] for icd in range(len(ref_compounds_dicts)) if matched_i2s[icd] is None]

    compound_index_pairs = dict([(i, matched_i2s[i]) for i in range(len(ref_compounds_dicts))])
    logger.info(f"found pairs of compounds: {compound_index_pairs}")
    report.index_match = compound_index_pairs

    field_stats_total = get_empty_field_change_stats()

    field_counter_ref_total = {ck: 0 for ck in list(CompoundFieldClass) + [None, ]}
    field_counter_act_total = {ck: 0 for ck in list(CompoundFieldClass) + [None, ]}

    compound_pair_deep_distances = []
    for i, j in compound_index_pairs.items():
        logger.info(f"inspecting a pair of compounds: {i} <-> {j}")
        ref_compound_dict = ref_compounds_dicts[i]
        if j is None:
            # TODO we may set act_compound_dict to be an empty dict...
            logger.warning(f"no matched act compound")
            continue

        act_compound_dict = act_compounds_dicts[j]
        deep_distance, field_stats_pair, field_counter_ref, field_counter_act = inspect_compound_pair(
            ref_compound_dict, act_compound_dict, prompt=prompt, skip_rule=skip_rule,
        )
        compound_pair_deep_distances.append(deep_distance)

        n_changed_fields_in_pair = 0
        for ck in field_stats_total:
            field_counter_ref_total[ck] += field_counter_ref[ck]
            logger.info(f"# of fields in ref: {ck} - {field_counter_ref[ck]}")
            field_counter_act_total[ck] += field_counter_act[ck]
            for fct in field_stats_total[ck]:
                logger.info(f"of which: {field_stats_pair[ck][fct]} - {fct}")
                field_stats_total[ck][fct] += field_stats_pair[ck][fct]
                n_changed_fields_in_pair += field_stats_pair[ck][fct]

        if n_changed_fields_in_pair:
            report.n_altered_compounds += 1
            logger.warning(
                f"# of CHANGED fields: {n_changed_fields_in_pair} -- the ref compound is CHANGED in this pair")
        else:
            logger.warning(
                f"# of CHANGED fields: {n_changed_fields_in_pair} -- the ref and act compounds are IDENTICAL in this pair")

    report.deep_distances = compound_pair_deep_distances
    report.field_change_stats = field_stats_total
    report.field_counter_ref = field_counter_ref_total
    report.field_counter_act = field_counter_act_total
    return report
