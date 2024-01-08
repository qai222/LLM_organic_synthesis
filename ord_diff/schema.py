from __future__ import annotations

import numpy as np
from deepdiff import DeepDiff
from google.protobuf import json_format
from ord_schema import reaction_pb2
from pydantic import BaseModel

from ord_diff.base import MessageType, DeltaType, CompoundLeafType
from ord_diff.utils import parse_deepdiff, flatten, find_best_match


# TODO for `Reaction.workups`, the order in the repeated field DOES matter,
#  so using `ignore_order` in deepdiff seems fishy.


def get_compound_leaf_type(leaf: Leaf):
    for ck in list(CompoundLeafType):
        if ck == CompoundLeafType.other:
            continue
        if ck in leaf.path_list:
            return CompoundLeafType(ck)
    return CompoundLeafType.other


def should_consider_leaf_in_nonstrict(leaf: Leaf, message_type: MessageType):
    if leaf.is_explicit:
        return True
    if message_type in (MessageType.COMPOUND, MessageType.PRODUCT_COMPOUND) and get_compound_leaf_type(
            leaf) != CompoundLeafType.other:
        return True
    if message_type == MessageType.REACTION_WORKUP:
        return True
    if message_type == MessageType.REACTION_CONDITIONS:
        return True
    return False


class Leaf(BaseModel):
    """ base model for a literal field """

    path_list: list[str | int]
    """ path to the leaf, which is a list of key/index """

    value: str | int | float
    """ literal value of this leaf """

    is_explicit: bool | None = None
    """ if this information is explicitly included in the text, only used for IE evaluation """

    @property
    def path_tuple(self):
        """ immutable path """
        return tuple(self.path_list)

    def __eq__(self, other: Leaf):
        return self.path_tuple == other.path_tuple

    def __hash__(self):
        return hash(self.path_tuple)

    class Config:
        validate_assignment = True


class MDict(BaseModel):
    """ base model for a message dictionary """

    type: MessageType
    """ predefined type of the message """

    d: dict
    """ the actual dictionary """

    leafs: list[Leaf]
    """ all leafs of this message """

    def is_type(self, message_type: MessageType):
        """ type checker """
        if message_type == MessageType.COMPOUND:
            mt = reaction_pb2.Compound
        elif message_type == MessageType.REACTION_WORKUP:
            mt = reaction_pb2.ReactionWorkup
        else:
            raise TypeError
        try:
            json_format.ParseDict(self.d, mt())
            return True
        except TypeError:
            return False

    def get_leaf(self, path: tuple[str | int, ...] | list[str | int]):
        """ access leaf directly """
        path_tuple = tuple([*path])
        for leaf in self.leafs:
            if path_tuple == leaf.path_tuple:
                return leaf
        raise KeyError

    @classmethod
    def from_dict(cls, message_dictionary: dict, message_type: MessageType, text_input: str | None = None):
        """ get message from a nested dictionary """
        leafs = []
        for path_tuple, value in flatten(message_dictionary).items():
            if text_input:
                # TODO not sure this is the best way...
                if str(value).lower() in text_input.lower():
                    explicit = True
                else:
                    explicit = False
            else:
                explicit = None
            leaf = Leaf(path_list=list(path_tuple), value=value, is_explicit=explicit)
            leafs.append(leaf)
        return cls(leafs=leafs, d=message_dictionary, type=message_type)

    @classmethod
    def from_message(cls, m, message_type: MessageType, text_input: str | None = None):
        d = json_format.MessageToDict(m)
        return MDict.from_dict(d, message_type, text_input)

    @property
    def compound_name(self):
        assert self.is_type(MessageType.COMPOUND)
        for identifier in self.d["identifiers"]:
            if identifier['type'] == 'NAME':
                return identifier['value']  # assuming only one name is defined
        raise ValueError('`NAME` not found in the compound dict')

    @property
    def workup_type(self):
        assert self.is_type(MessageType.REACTION_WORKUP)
        try:
            t = self.d['type']
            return t
        except KeyError:
            raise ValueError('failed to access the `type` field of a ReactionWorkup')


class MDictDiff(BaseModel):
    """ base model for a comparison """

    md1: MDict
    """ first message """

    md2: MDict
    """ second message """

    delta_leafs: dict[DeltaType, list[Leaf]]
    """
    a dictionary showing how leafs are changed, unchanged leafs are not included
    ADDITION -> added leafs from m2
    REMOVAL -> removed leafs from m1
    ALTERATION -> altered leafs from m1
    """

    delta_paths: dict[DeltaType, list[tuple[str | int, ...]]]
    """
    similar to `delta_leafs`, except the values here are paths to a generic field (may not be a leaf)
    """

    deep_distance: float
    """ deep distance between m1 and m2 """

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True

    @property
    def delta_leafs_nonstrict(self):
        return {
            k: [vv for vv in v if should_consider_leaf_in_nonstrict(vv, self.md1.type)]
            for k, v in self.delta_leafs.items()
        }

    def is_altered(self, strict: bool):
        if strict:
            return self.deep_distance > 0
        else:
            n_altered_leaf = 0
            for v in self.delta_leafs_nonstrict.values():
                n_altered_leaf += len(v)
            return n_altered_leaf > 0

    @classmethod
    def from_md_pair(cls, md1: MDict, md2: MDict):
        assert md1.type == md2.type

        dd = DeepDiff(
            md1.d, md2.d,
            ignore_order=True, verbose_level=2, view='tree', get_deep_distance=True
        )

        (
            deep_distance,
            paths_added, paths_removed, paths_altered_1, paths_altered_2,
            leaf_paths_added, leaf_paths_removed, leaf_paths_altered_1, leaf_paths_altered_2,
        ) = parse_deepdiff(dd)

        delta_leafs = {
            DeltaType.ADDITION: [md2.get_leaf(p) for p in leaf_paths_added],
            DeltaType.REMOVAL: [md1.get_leaf(p) for p in leaf_paths_removed],
            DeltaType.ALTERATION: [md1.get_leaf(p) for p in leaf_paths_altered_1],
        }

        delta_paths = {
            DeltaType.ADDITION: paths_added,
            DeltaType.REMOVAL: paths_removed,
            DeltaType.ALTERATION: paths_altered_1
        }

        return cls(
            md1=md1,
            md2=md2,
            deep_distance=deep_distance,
            delta_paths=delta_paths,
            delta_leafs=delta_leafs,
        )

    @classmethod
    def from_message_pair(cls, m1, m2, message_type: MessageType, text1: str = None, text2: str = None):
        return MDictDiff.from_md_pair(
            md1=MDict.from_message(m1, message_type, text1),
            md2=MDict.from_message(m2, message_type, text2),
        )


class MDictListDiff(BaseModel):
    """ base model for a comparison between two lists of messages """

    md1_list: list[MDict]
    """ the first list of messages """

    md2_list: list[MDict]
    """ the second list of messages """

    index_match: dict[int, int | None]
    """
    index_match[i] = <the matched index of md2> | None if no match
    
    a list of md2 indices is a match if
    1. no repeating index, unless None
    2. size = md1_list
    3. minimize the sum of deep distance
    """

    pair_comparisons: list[MDictDiff | None]
    """
    for each match, if the matched md2 is not None, a comparison is made
    note the num of pairs == len(m1_list)
    """

    n_changed_strict: int
    """ num of md pairs that have at least one field added/removed/changed """

    n_changed_nonstrict: int

    @property
    def n_md1(self):
        """ number of messages in the md1 list """
        return len(self.md1_list)

    @property
    def n_md2(self):
        """ number of messages in the md2 list """
        return len(self.md2_list)

    @property
    def n_absent(self):
        """ messages that are absent in the md2 list but present in the md1 list """
        n = self.n_md1 - self.n_md2
        if n > 0:
            return n
        else:
            return 0

    @property
    def n_excess(self):
        """ messages in the md2 list that has no match to a md1 message """
        n = self.n_md2 - self.n_md1
        if n > 0:
            return n
        else:
            return 0

    @classmethod
    def from_message_list_pair(
            cls, m1_list, m2_list, message_type: MessageType,
            m1_text: str = None, m2_text: str = None
    ):
        md1_list = [MDict.from_message(m, message_type, m1_text) for m in m1_list]
        md2_list = [MDict.from_message(m, message_type, m2_text) for m in m2_list]
        return MDictListDiff.from_md_list_pair(md1_list, md2_list)

    @classmethod
    def from_md_list_pair(
            cls,
            md1_list: list[MDict],
            md2_list: list[MDict],
    ):
        """
        find the differences between two lists of messages
        1. each message in ref_list is matched with one from act_list, matched with None if missing
        2. use deepdiff to inspect matched pairs
        """
        assert len(md1_list) and len(md2_list)
        assert len(set([md.type for md in md1_list])) == 1
        assert len(set([md.type for md in md2_list])) == 1
        assert md1_list[0].type == md2_list[0].type

        message_type = md1_list[0].type
        matched_i2s = MDictListDiff.get_index_match(md1_list, md2_list, message_type=message_type)

        pair_comparisons = []
        n_changed_strict = 0
        n_changed_nonstrict = 0
        for i, m1 in enumerate(md1_list):
            j = matched_i2s[i]
            if j is None:
                pair_comparison = None
            else:
                m2 = md2_list[j]
                pair_comparison = MDictDiff.from_md_pair(m1, m2)
                if pair_comparison.is_altered(strict=True):
                    n_changed_strict += 1
                if pair_comparison.is_altered(strict=False):
                    n_changed_nonstrict += 1
            pair_comparisons.append(pair_comparison)

        return cls(
            md1_list=md1_list,
            md2_list=md2_list,
            pair_comparisons=pair_comparisons,
            n_changed_strict=n_changed_strict,
            n_changed_nonstrict=n_changed_nonstrict,
            index_match=matched_i2s,
        )

    @staticmethod
    def index_match_distance_matrix(m1_list: list[MDict], m2_list: list[MDict], message_type: MessageType):
        assert len(m1_list) and len(m2_list)

        indices1 = [*range(len(m1_list))]
        indices2 = [*range(len(m2_list))]

        dist_mat = np.zeros((len(indices1), len(indices2)))
        for i1 in indices1:
            md1 = m1_list[i1]
            if message_type == MessageType.COMPOUND:
                md1_id = md1.compound_name
            elif message_type == MessageType.REACTION_WORKUP:
                md1_id = md1.workup_type
            else:
                md1_id = None
            for i2 in indices2:
                assert isinstance(i2, int)
                md2 = m2_list[i2]
                if message_type == MessageType.COMPOUND:
                    md2_id = md2.compound_name
                elif message_type == MessageType.REACTION_WORKUP:
                    md2_id = md2.workup_type
                else:
                    md2_id = None
                try:
                    distance_id = DeepDiff(md1_id, md2_id, get_deep_distance=True).to_dict()['deep_distance']
                except KeyError:
                    distance_id = 0
                try:
                    distance_full = DeepDiff(
                        md1.d, md2.d, get_deep_distance=True, ignore_order=True
                    ).to_dict()['deep_distance']
                except KeyError:
                    distance_full = 0
                distance = distance_id * 100 + distance_full  # large penalty for wrong names
                dist_mat[i1][i2] = distance
        return dist_mat

    @staticmethod
    def get_index_match(m1_list: list[MDict], m2_list: list[MDict], message_type: MessageType):
        """
        for each compound in m1, find the most similar one in m2 based on a weighted deep distance,
        use the full deep distance to break tie

        :param m1_list:
        :param m2_list:
        :param message_type:
        :return:
        """

        indices1 = [*range(len(m1_list))]
        indices2 = [*range(len(m2_list))]

        distance_matrix = MDictListDiff.index_match_distance_matrix(m1_list, m2_list, message_type)

        while len(indices2) < len(indices1):
            indices2.append(None)

        return find_best_match(indices1, indices2, distance_matrix)
