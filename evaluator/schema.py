from __future__ import annotations

import json
from copy import deepcopy
from enum import Enum

from loguru import logger
from ord_schema.message_helpers import json_format, find_submessages, Type, MessageType
from ord_schema.proto import reaction_pb2
from pandas._typing import FilePath
from pydantic import BaseModel
from tqdm import tqdm

from evaluator.ord_deepdiff import *


class ModelOuputError(Exception):
    pass


class ModelOutput:

    def __init__(self, identifier: str, raw: str, prompt: str = None, response: str = None, ref: str = None):
        """
        the full output of a model should be
        `prompt_header` + `prompt` + `response_header` + `response`
        there can be arbitrary number of line breaks between any two of them
        """
        self.identifier = identifier
        self.raw = raw
        self.prompt = prompt
        self.response = response
        self.ref = ref

    def as_dict(self):
        return {
            "identifier": self.identifier,
            "prompt": self.prompt,
            "response": self.response,
            "reference": self.ref
        }

    @staticmethod
    def parse_raw(
            raw,
            prompt_template="### Procedure:\n{instruction}\n\n### ORD-JSON:\n",
            prompt_header="### Procedure:",
            response_header="### ORD-JSON:",
    ):
        assert prompt_header in raw
        assert response_header in raw
        n_line_breaks = prompt_template.count("\n")
        assert n_line_breaks == 4
        assert raw.count("\n") == n_line_breaks

        p_header, prompt, _, r_header, response = raw.split("\n")
        # try:
        #     p_header, prompt, _, r_header, response = raw.split("\n")
        # except Exception as e:
        #     raise ModelOuputError(f"invalid text: {raw}")
        assert p_header == prompt_header
        assert r_header == response_header
        return prompt, response

    @classmethod
    def from_raw_alpaca(
            cls, raw: str, ref: str, identifier: str,
            prompt_template="### Procedure:\n{instruction}\n\n### ORD-JSON:\n",
            prompt_header="### Procedure:",
            response_header="### ORD-JSON:",
    ):
        """ create from responses to alpaca-like instruction prompts """
        try:
            prompt, response = ModelOutput.parse_raw(raw, prompt_template, prompt_header, response_header)
        except Exception:
            raise ModelOuputError
        model_output = cls(identifier, raw, prompt, response, ref)
        return model_output


class OrdMajorField(str, Enum):
    inputs = "inputs"
    outcomes = "outcomes"
    workups = "workups"
    conditions = "conditions"


class EvaluatorError(Exception):
    pass


class EvaluatorReport(BaseModel):
    # inputs
    inputs_compound_change_stats: dict[FieldChangeType, int] = {k: 0 for k in list(FieldChangeType)}
    inputs_compound_n_ref_compounds: int = 0

    inputs_compound_field_change_stats: compound_field_change_stats_type = get_empty_field_change_stats()
    inputs_compound_field_counter_ref: dict[CompoundFieldClass | None, int] = {k: 0 for k in
                                                                               list(CompoundFieldClass) + [None, ]}

    inputs_compound_lol_n_misplaced_groups: int = 0

    # outcomes
    outcomes_compound_change_stats: dict[FieldChangeType, int] = {k: 0 for k in list(FieldChangeType)}
    outcomes_compound_n_ref_compounds: int = 0

    outcomes_compound_field_change_stats: compound_field_change_stats_type = get_empty_field_change_stats()
    outcomes_compound_field_counter_ref: dict[CompoundFieldClass | None, int] = {k: 0 for k in
                                                                                 list(CompoundFieldClass) + [None, ]}

    outcomes_compound_lol_n_misplaced_groups: int = 0

    # workups
    workups_n_workups_absent: int = 0
    workups_n_workups_excess: int = 0
    workups_n_workups_ref: int = 0

    # conditions
    conditions_n_erroneous_condition_types: int = 0
    conditions_n_ref_conditions: int = 0

    def __add__(self, other: EvaluatorReport) -> EvaluatorReport:

        assert isinstance(other, EvaluatorReport)
        dict_self = self.model_dump()
        dict_other = other.model_dump()
        result = EvaluatorReport()

        for k, v1 in dict_self.items():
            v2 = dict_other[k]
            if isinstance(v1, int):
                setattr(result, k, v1 + v2)
            elif isinstance(v1, dict):
                result_v = deepcopy(v1)
                v1_nested_level = get_dict_depth(v1)
                if v1_nested_level == 1:
                    for kk in v1:
                        result_v[kk] += v2[kk]
                elif v1_nested_level == 2:
                    for kk1 in v1:
                        for kk2 in v1[kk1]:
                            result_v[kk1][kk2] += v2[kk1][kk2]
                else:
                    raise EvaluatorError(
                        f"the field: {k} of class: {self.__class__.__name__} is a dict: {v1}, "
                        f"it has a level of: {v1_nested_level} which is not expected:"
                    )
                setattr(result, k, result_v)
        return result


class Evaluator:

    def __init__(self, output: ModelOutput):
        self.output = output

        try:
            self.reaction_dict = json.loads(self.output.response)
        except Exception as e:
            raise EvaluatorError(str(e))

        try:
            self.reaction_message = json_format.Parse(self.output.response, reaction_pb2.Reaction())
        except Exception as e:
            raise EvaluatorError(str(e))

        self.reaction_dict_ref = json.loads(self.output.ref)
        self.reaction_message_ref = json_format.Parse(self.output.ref, reaction_pb2.Reaction())

    def find_messages(
            self, in_field: OrdMajorField | None = None,
            message_type: Type[MessageType] = reaction_pb2.Compound, ref=False
    ):
        """ find ord messages in a specific part of reaction """
        if ref:
            reaction_message = self.reaction_message_ref
        else:
            reaction_message = self.reaction_message

        reaction_message: reaction_pb2.Reaction

        messages = []
        if in_field is None:
            messages += find_submessages(reaction_message, message_type)
        elif in_field == 'conditions':
            messages += find_submessages(reaction_message.conditions, message_type)
        elif in_field in ('outcomes', 'workups'):
            for m in getattr(reaction_message, in_field):
                messages += find_submessages(m, message_type)
        elif in_field == 'inputs':
            for v in getattr(reaction_message, in_field).values():
                messages += find_submessages(v, message_type)
        return messages

    def get_diff_report(
            self,
            kind: DiffReportKind,
            in_field: OrdMajorField,
    ) -> DiffReportListOfCompounds | DiffReportCompoundLol | DiffReportListOfReactionWorkups | DiffReportReactionConditions:
        """
        get an individual diff report

        :param kind:
        :param in_field: does not matter for workups and conditions
        :return:
        """
        if kind == DiffReportKind.LIST_OF_COMPOUNDS:
            if in_field == "outcomes":
                compound_class = reaction_pb2.ProductCompound
            elif in_field == "inputs":
                compound_class = reaction_pb2.Compound
            else:
                raise EvaluatorError(f"not allowed report for: {kind} in the field of: {in_field}")
            ref_compounds = self.find_messages(in_field=in_field, message_type=compound_class, ref=True)
            act_compounds = self.find_messages(in_field=in_field, message_type=compound_class, ref=False)
            return diff_list_of_compounds(ref_compounds, act_compounds)

        elif kind == DiffReportKind.LIST_OF_COMPOUND_LISTS:
            if in_field == "outcomes":
                ref_lol = [ri.products for ri in self.reaction_message_ref.outcomes]
                act_lol = [ri.products for ri in self.reaction_message.outcomes]
            elif in_field == "inputs":
                ref_lol = [ri.components for ri in list(self.reaction_message_ref.inputs.values())]
                act_lol = [ri.components for ri in list(self.reaction_message.inputs.values())]
            else:
                raise EvaluatorError(f"not allowed report for: {kind} in the field of: {in_field}")
            return diff_list_of_compound_lists(ref_lol, act_lol)

        elif kind == DiffReportKind.REACTION_CONDITIONS:
            ref_conditions = self.reaction_message.conditions
            act_conditions = self.reaction_message_ref.conditions
            return diff_reaction_conditions(ref_conditions, act_conditions)

        elif kind == DiffReportKind.LIST_OF_REACTION_WORKUPS:
            ref_workups = self.reaction_message_ref.workups
            act_workups = self.reaction_message.workups
            return diff_list_of_reaction_workups(ref_workups, act_workups)

    def run_evaluate(self) -> EvaluatorReport:
        report = EvaluatorReport()

        # inputs
        r_inputs_list_of_compound = self.get_diff_report(
            kind=DiffReportKind.LIST_OF_COMPOUNDS, in_field=OrdMajorField.inputs
        )
        report.inputs_compound_change_stats = r_inputs_list_of_compound.compound_change_stats
        report.inputs_compound_n_ref_compounds = r_inputs_list_of_compound.n_ref_compounds
        report.inputs_compound_field_change_stats = r_inputs_list_of_compound.field_change_stats
        report.inputs_compound_field_counter_ref = r_inputs_list_of_compound.field_counter_ref

        r_inputs_compound_lol = self.get_diff_report(
            kind=DiffReportKind.LIST_OF_COMPOUND_LISTS, in_field=OrdMajorField.inputs
        )
        report.inputs_compound_lol_n_misplaced_groups = r_inputs_compound_lol.n_misplaced_groups

        # outcomes
        r_outcomes_list_of_compound = self.get_diff_report(
            kind=DiffReportKind.LIST_OF_COMPOUNDS, in_field=OrdMajorField.outcomes
        )
        report.outcomes_compound_change_stats = r_outcomes_list_of_compound.compound_change_stats
        report.outcomes_compound_n_ref_compounds = r_outcomes_list_of_compound.n_ref_compounds
        report.outcomes_compound_field_change_stats = r_outcomes_list_of_compound.field_change_stats
        report.outcomes_compound_field_counter_ref = r_outcomes_list_of_compound.field_counter_ref

        r_outcomes_compound_lol = self.get_diff_report(
            kind=DiffReportKind.LIST_OF_COMPOUND_LISTS, in_field=OrdMajorField.outcomes
        )
        report.outcomes_compound_lol_n_misplaced_groups = r_outcomes_compound_lol.n_misplaced_groups

        # workups
        r_workups = self.get_diff_report(kind=DiffReportKind.LIST_OF_REACTION_WORKUPS, in_field=OrdMajorField.workups)
        report.workups_n_workups_absent = r_workups.n_workups_absent
        report.workups_n_workups_excess = r_workups.n_workups_excess
        report.workups_n_workups_ref = r_workups.n_workups_ref

        # conditions
        r_conditions = self.get_diff_report(kind=DiffReportKind.REACTION_CONDITIONS, in_field=OrdMajorField.conditions)
        report.conditions_n_erroneous_condition_types = r_conditions.n_erroneous_condition_types
        report.conditions_n_ref_conditions = r_conditions.n_ref_conditions

        return report

    @staticmethod
    def evaluate_model_outputs(model_outputs: list[ModelOutput], ):
        report_dictionary = dict()
        report_dictionary: dict[str, EvaluatorReport | None]
        report_summary = EvaluatorReport()

        for model_output in tqdm(model_outputs, desc="evaluate model outputs"):
            try:
                inference_evaluator = Evaluator(model_output)
            except EvaluatorError:
                report = None
                report_dictionary[model_output.identifier] = report
                logger.warning(f"invalid JSON/ORD: {model_output.identifier}")
                continue
            report = inference_evaluator.eval()
            report_summary += report
        return dict(report_dictionary=report_dictionary, report_summary=report_summary)

    @staticmethod
    def evaluators_from_json(json_file: FilePath, first_k=None) -> tuple[list[Evaluator], int]:
        with open(json_file, "r") as f:
            data = json.load(f)
        if first_k:
            data = data[:first_k]
        evals = []
        n_invalid = 0
        for record in tqdm(data, desc="load json to evaluators"):
            model_output = ModelOutput.from_raw_alpaca(
                raw=record['response'],
                ref=record['output'],
                identifier=record['reaction_id']
            )
            try:
                inference_evaluator = Evaluator(model_output)
            except EvaluatorError:
                logger.warning(f"invalid JSON/ORD: {model_output.identifier}")
                n_invalid += 1
                continue
            evals.append(inference_evaluator)
        logger.warning(f"total invalid JSON/ORD: {n_invalid}")
        return evals, n_invalid

    @staticmethod
    def evaluate_llama_inference_json(json_file: FilePath, first_k=None):
        evaluators, n_invalid = Evaluator.evaluators_from_json(json_file, first_k)
        report_dictionary = dict()
        report_dictionary: dict[str, EvaluatorReport | None]
        report_summary = EvaluatorReport()
        for evaluator in tqdm(evaluators, desc="evaluating"):
            report = evaluator.run_evaluate()
            report_dictionary[evaluator.output.identifier] = report.model_dump()
            report_summary += report
        return dict(report_dictionary=report_dictionary, report_summary=report_summary.model_dump(),
                    n_invalid=n_invalid)
