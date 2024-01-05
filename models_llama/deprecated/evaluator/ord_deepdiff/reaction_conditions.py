from __future__ import annotations

from deepdiff import DeepDiff
from deepdiff.helper import NotPresent
from deepdiff.model import DiffLevel, PrettyOrderedSet
from google.protobuf import json_format
from ord_schema.proto import reaction_pb2

from models_llama.deprecated.evaluator.ord_deepdiff.base import DiffReport, DiffReportKind, FieldChangeType
from models_llama.deprecated.evaluator.ord_deepdiff.utils import DeepDiffKey


class DiffReportReactionConditions(DiffReport):
    kind: DiffReportKind = DiffReportKind.REACTION_CONDITIONS

    erroneous_condition_types: list[str] = []

    condition_type_stats: dict[FieldChangeType, int] = {fc: 0 for fc in list(FieldChangeType)}

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


def diff_reaction_conditions(c1: reaction_pb2.ReactionConditions,
                             c2: reaction_pb2.ReactionConditions) -> DiffReportReactionConditions:
    report = DiffReportReactionConditions()

    cd1 = json_format.MessageToDict(c1)
    cd2 = json_format.MessageToDict(c2)

    report.reference = cd1
    report.actual = cd2

    condition_type_stats = {fc: 0 for fc in list(FieldChangeType)}
    deep_distance = 0.0
    diff = DeepDiff(cd1, cd2, ignore_order=True, view='tree', get_deep_distance=True)
    for k, v in diff.to_dict().items():
        k: str
        v: PrettyOrderedSet[DiffLevel] | float
        if k == DeepDiffKey.deep_distance.value:
            deep_distance = v
        else:
            for value_changed_level in v:
                t1 = value_changed_level.t1
                t2 = value_changed_level.t2
                is_ref_none = isinstance(t1, NotPresent)
                is_act_none = isinstance(t2, NotPresent)
                path_list = value_changed_level.path(output_format='list')
                condition_type = path_list[0]
                if is_ref_none and not is_act_none:
                    fct = FieldChangeType.ADDITION
                elif not is_ref_none and is_act_none:
                    fct = FieldChangeType.REMOVAL
                elif not is_ref_none and not is_act_none:
                    fct = FieldChangeType.ALTERATION
                else:
                    raise ValueError
                condition_type_stats[fct] += 1

    report.condition_type_stats = condition_type_stats
    report.deep_distance = deep_distance
    return report
