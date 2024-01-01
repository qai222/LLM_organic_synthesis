from __future__ import annotations

import json
import math
import pprint
import textwrap
from copy import deepcopy
from enum import Enum

import pandas as pd
from loguru import logger
from ord_schema.message_helpers import json_format, find_submessages, Type, MessageType
from ord_schema.proto import reaction_pb2
from pandas._typing import FilePath
from pydantic import BaseModel
from tqdm import tqdm
from evaluator.ord_deepdiff import *
import signal
from functools import wraps


def timeout(seconds, default=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def signal_handler(signum, frame):
                raise TimeoutError("Timed out!")

            # Set up the signal handler for timeout
            signal.signal(signal.SIGALRM, signal_handler)

            # Set the initial alarm for the integer part of seconds
            signal.setitimer(signal.ITIMER_REAL, seconds)

            try:
                result = func(*args, **kwargs)
            except TimeoutError:
                return default
            finally:
                signal.alarm(0)

            return result

        return wrapper

    return decorator

class ModelOuputError(Exception):
    pass


class ModelOutput:

    def __init__(self, identifier: str, raw: str, prompt: str = None, response: str = None, ref: str = None, instruction: str = None):
        """
        the full output of a model should be
        `prompt_header` + `prompt` + `response_header` + `response`
        there can be arbitrary number of line breaks between any two of them
        """
        self.instruction = instruction
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
            instruction:str=None,
    ):
        """ create from responses to alpaca-like instruction prompts """
        try:
            prompt, response = ModelOutput.parse_raw(raw, prompt_template, prompt_header, response_header)
        except Exception:
            raise ModelOuputError
        model_output = cls(identifier, raw, prompt, response, ref, instruction)
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
    workups_n_workups_altered: int = 0
    workups_n_workups_ref: int = 0

    # conditions
    conditions_condition_type_stats: dict[FieldChangeType, int] = {k: 0 for k in list(FieldChangeType)}
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

    def get_table_compound_fields(self, from_inputs=True):
        """ a more readable table at the field level """
        if from_inputs:
            field_change_stats = self.inputs_compound_field_change_stats
            field_counter_ref = self.inputs_compound_field_counter_ref
        else:
            field_change_stats = self.outcomes_compound_field_change_stats
            field_counter_ref = self.outcomes_compound_field_counter_ref

        records = []
        for fc in field_change_stats:
            if fc is None:
                continue
            for fct in field_change_stats[fc]:
                n_changed = field_change_stats[fc][fct]
                n_ref = field_counter_ref[fc]
                try:
                    percent = n_changed / n_ref
                except ZeroDivisionError:
                    percent = math.nan
                record = {
                    "Field Category": fc.value,
                    "Change Type": fct.value,
                    "fields": "{}/{} {:.2%}".format(n_changed, n_ref, percent)
                }
                records.append(record)
        df = pd.DataFrame.from_records(records)
        return df.pivot(index="Field Category", columns="Change Type", values="fields")

    def get_table_messages(self):
        """ a more readable table at the sub message level """

        def format_cell(n, d):
            try:
                r = n / d
            except ZeroDivisionError:
                r = 0
            return "{}/{} {:.2%}".format(n, d, r)

        row_inputs_compound = dict(message="Compounds (inputs)")
        row_outputs_compound = dict(message="Compounds (outcomes)")
        row_workups = dict(message="workups")
        row_conditions = dict(message="conditions")
        for fc in list(FieldChangeType):
            row_inputs_compound[fc.value] = format_cell(
                self.inputs_compound_change_stats[fc],
                self.inputs_compound_n_ref_compounds,
            )
            row_outputs_compound[fc.value] = format_cell(
                self.outcomes_compound_change_stats[fc],
                self.outcomes_compound_n_ref_compounds,
            )
            row_conditions[fc.value] = format_cell(
                self.conditions_condition_type_stats[fc],
                self.conditions_n_ref_conditions,
            )
            if fc == FieldChangeType.ADDITION:
                row_workups[fc.value] = format_cell(
                    self.workups_n_workups_excess,
                    self.workups_n_workups_ref,
                )
            elif fc == FieldChangeType.REMOVAL:
                row_workups[fc.value] = format_cell(
                    self.workups_n_workups_absent,
                    self.workups_n_workups_ref,
                )
            elif fc == FieldChangeType.ALTERATION:
                row_workups[fc.value] = format_cell(
                    self.workups_n_workups_altered,
                    self.workups_n_workups_ref,
                )
        rows = [
            row_inputs_compound,
            row_outputs_compound,
            row_workups,
            row_conditions,
        ]
        df = pd.DataFrame.from_records(rows)
        return df


class Evaluator:

    def __init__(self, output: ModelOutput,
                 skip_rule: FieldSkipRule = FieldSkipRule.ignore_absent_in_prompt_with_exceptions):
        self.skip_rule = skip_rule
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
        logger.info(f"getting diff report for: {kind} in_field: {in_field}")
        if kind == DiffReportKind.LIST_OF_COMPOUNDS:
            if in_field == "outcomes":
                compound_class = reaction_pb2.ProductCompound
            elif in_field == "inputs":
                compound_class = reaction_pb2.Compound
            else:
                raise EvaluatorError(f"not allowed report for: {kind} in the field of: {in_field}")
            ref_compounds = self.find_messages(in_field=in_field, message_type=compound_class, ref=True)
            act_compounds = self.find_messages(in_field=in_field, message_type=compound_class, ref=False)
            r = diff_list_of_compounds(ref_compounds, act_compounds, self.output.prompt, skip_rule=self.skip_rule)

        elif kind == DiffReportKind.LIST_OF_COMPOUND_LISTS:
            if in_field == "outcomes":
                ref_lol = [ri.products for ri in self.reaction_message_ref.outcomes]
                act_lol = [ri.products for ri in self.reaction_message.outcomes]
            elif in_field == "inputs":
                ref_lol = [ri.components for ri in list(self.reaction_message_ref.inputs.values())]
                act_lol = [ri.components for ri in list(self.reaction_message.inputs.values())]
            else:
                raise EvaluatorError(f"not allowed report for: {kind} in the field of: {in_field}")
            r = diff_list_of_compound_lists(ref_lol, act_lol)

        elif kind == DiffReportKind.REACTION_CONDITIONS:
            ref_conditions = self.reaction_message.conditions
            act_conditions = self.reaction_message_ref.conditions
            r = diff_reaction_conditions(ref_conditions, act_conditions)

        elif kind == DiffReportKind.LIST_OF_REACTION_WORKUPS:
            ref_workups = self.reaction_message_ref.workups
            act_workups = self.reaction_message.workups
            r = diff_list_of_reaction_workups(ref_workups, act_workups)

        else:
            raise ValueError

        logger.info("report obtained")
        return r

    @timeout(1)
    def run_evaluate(self) -> EvaluatorReport:
        report = EvaluatorReport()

        logger.info(f">> START INSPECTING: {self.output.identifier}")
        try:
            logger.info(f"PROMPT is:\n{textwrap.fill(self.output.prompt, 140)}")
        except AttributeError:
            logger.info(f"PROMPT is:\n{self.output.prompt}")

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
        report.workups_n_workups_altered = r_workups.n_workups_altered

        # conditions
        r_conditions = self.get_diff_report(kind=DiffReportKind.REACTION_CONDITIONS, in_field=OrdMajorField.conditions)
        report.conditions_condition_type_stats = r_conditions.condition_type_stats
        report.conditions_n_ref_conditions = r_conditions.n_ref_conditions

        logger.info(pprint.pformat(report.model_dump()))
        logger.info(f">> FINISH INSPECTING\n")
        return report

    @staticmethod
    def evaluators_from_json(
            json_file: FilePath, slice_indices: tuple[int, int] | None = None,
            skip_rule: FieldSkipRule = FieldSkipRule.ignore_absent_in_prompt_with_exceptions,
    ) -> tuple[list[Evaluator], int]:
        with open(json_file, "r") as f:
            data = json.load(f)
        if slice_indices:
            data = data[slice_indices[0]:slice_indices[1]]
        evals = []
        n_invalid = 0
        for record in tqdm(data, desc="load json to evaluators"):
            model_output = ModelOutput.from_raw_alpaca(
                raw=record['response'],
                ref=record['output'],
                identifier=record['reaction_id']
            )
            try:
                inference_evaluator = Evaluator(model_output, skip_rule=skip_rule)
            except EvaluatorError:
                logger.warning(f"invalid JSON/ORD: {model_output.identifier}")
                n_invalid += 1
                continue
            evals.append(inference_evaluator)
        logger.warning(f"total invalid JSON/ORD: {n_invalid}")
        return evals, n_invalid

    @staticmethod
    def evaluate_llama_inference_json(
            json_file: FilePath, slice_indices: tuple[int, int] | None = None,
            skip_rule: FieldSkipRule = FieldSkipRule.ignore_absent_in_prompt_with_exceptions,
    ):
        evaluators, n_invalid = Evaluator.evaluators_from_json(json_file, slice_indices=slice_indices,
                                                               skip_rule=skip_rule)
        report_dictionary = dict()
        report_dictionary: dict[str, EvaluatorReport | None]
        report_summary = EvaluatorReport()
        for evaluator in tqdm(evaluators, desc="evaluating"):
            report = evaluator.run_evaluate()
            report_dictionary[evaluator.output.identifier] = report.model_dump()
            report_summary += report
        return dict(report_dictionary=report_dictionary, report_summary=report_summary,
                    n_invalid=n_invalid)
