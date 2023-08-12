from __future__ import annotations

from deepdiff import DeepDiff
from deepdiff.model import DiffLevel, PrettyOrderedSet
from google.protobuf import json_format
from ord_schema.proto import reaction_pb2

from evaluator.ord_deepdiff.base import DiffReport, DiffReportKind
from evaluator.ord_deepdiff.utils import DeepDiffKey


class DiffReportReactionConditions(DiffReport):
    kind: DiffReportKind = DiffReportKind.REACTION_CONDITIONS

    erroneous_condition_types: list[str] = []

    @property
    def n_erroneous_condition_types(self):
        return len(self.erroneous_condition_types)

    @property
    def n_ref_conditions(self):
        return len(self.reference)

    @property
    def n_act_conditions(self):
        return len(self.actual)

    deep_distance: float = 0.0


def diff_reaction_conditions(c1: reaction_pb2.ReactionConditions, c2: reaction_pb2.ReactionConditions):
    report = DiffReportReactionConditions()

    cd1 = json_format.MessageToDict(c1)
    cd2 = json_format.MessageToDict(c2)

    report.reference = cd1
    report.actual = cd2

    erroneous_condition_types = []
    deep_distance = 0.0
    diff = DeepDiff(cd1, cd2, ignore_order=True, view='tree', get_deep_distance=True)
    for k, v in diff.to_dict().items():
        k: str
        v: PrettyOrderedSet[DiffLevel] | float
        if k == DeepDiffKey.deep_distance.value:
            deep_distance = v
        else:
            for value_changed_level in v:
                path_list = value_changed_level.path(output_format='list')
                condition_type = path_list[0]
                erroneous_condition_types.append(condition_type)
    report.erroneous_condition_types = erroneous_condition_types
    report.deep_distance = deep_distance
    return report
