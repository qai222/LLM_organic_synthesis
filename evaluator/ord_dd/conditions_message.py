from __future__ import annotations

from deepdiff import DeepDiff
from deepdiff.model import DiffLevel, PrettyOrderedSet
from google.protobuf import json_format
from ord_schema.proto import reaction_pb2
from pydantic import BaseModel

from evaluator.ord_dd.utils import DeepDiffKey


class ConditionsDiffReport(BaseModel):
    erroneous_condition_types: list[str] = []

    n_ref_conditions: int = 0

    n_act_conditions: int = 0

    deep_distance: float = 0.0  # not averaged

    class Config:
        validate_assignment = True


def diff_conditions(c1: reaction_pb2.ReactionConditions, c2: reaction_pb2.ReactionConditions):
    report = ConditionsDiffReport()

    cd1 = json_format.MessageToDict(c1)
    cd2 = json_format.MessageToDict(c2)

    report.n_ref_conditions = len(cd1)
    report.n_act_conditions = len(cd2)

    erroneous_condition_types = []
    deep_distance = 0.0
    diff = DeepDiff(cd1, cd2, ignore_order=True, view='tree', get_deep_distance=True)
    for k, v in diff.to_dict().items():
        k: str
        v: PrettyOrderedSet[DiffLevel] | float
        if k == DeepDiffKey.deep_distance.value:
            deep_distance = v
        else:  # this means we include all add/remove/change
            for value_changed_level in v:
                path_list = value_changed_level.path(output_format='list')
                condition_type = path_list[0]
                erroneous_condition_types.append(condition_type)
    report.erroneous_condition_types = erroneous_condition_types
    report.deep_distance = deep_distance
    return report
