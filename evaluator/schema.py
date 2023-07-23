from __future__ import annotations

import json
from typing import Literal

import pandas as pd
from tqdm import tqdm
from ord_schema.message_helpers import json_format, find_submessages, Type, MessageType
from ord_schema.proto import reaction_pb2

from evaluator.ord_dd import *


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

    @classmethod
    def from_raw_alpaca(
            cls, raw: str, ref: str, identifier: str,
            prompt_template="### Procedure:\n{instruction}\n\n### ORD-JSON:\n",
            prompt_header="### Procedure:",
            response_header="### ORD-JSON:",
    ):
        """ create from responses to alpaca-like instruction prompts """
        assert prompt_header in raw
        assert response_header in raw
        n_line_breaks = prompt_template.count("\n")
        assert n_line_breaks == 4
        assert raw.count("\n") == n_line_breaks

        try:
            p_header, prompt, _, r_header, response = raw.split("\n")
        except Exception as e:
            raise ValueError(f"invalid text: {raw}")
        assert p_header == prompt_header
        assert r_header == response_header
        model_output = cls(identifier, raw, prompt, response, ref)
        return model_output


class EvaluatorSniffError(Exception):
    pass


class Evaluator:

    def __init__(self, output: ModelOutput):
        self.output = output

        try:
            self.reaction_dict = json.loads(self.output.response)
        except Exception as e:
            raise EvaluatorSniffError(str(e))

        try:
            self.reaction_message = json_format.Parse(self.output.response, reaction_pb2.Reaction())
        except Exception as e:
            raise EvaluatorSniffError(str(e))

        self.reaction_dict_ref = json.loads(self.output.ref)
        self.reaction_message_ref = json_format.Parse(self.output.ref, reaction_pb2.Reaction())

    def find_messages(
            self, key: Literal['inputs', 'outcomes', 'workups', 'conditions', None] = None,
            message_type: Type[MessageType] = reaction_pb2.Compound, ref=False
    ):
        """ find ord messages in a specific part of reaction """
        if ref:
            reaction_message = self.reaction_message_ref
        else:
            reaction_message = self.reaction_message

        reaction_message: reaction_pb2.Reaction

        messages = []
        if key is None:
            messages += find_submessages(reaction_message, message_type)
        elif key == 'conditions':
            messages += find_submessages(reaction_message.conditions, message_type)
        elif key in ('outcomes', 'workups'):
            for m in getattr(reaction_message, key):
                messages += find_submessages(m, message_type)
        elif key == 'inputs':
            for v in getattr(reaction_message, key).values():
                messages += find_submessages(v, message_type)
        return messages

    def evaluate_inputs_compounds_list(self) -> CompoundListDiffReport:
        """ compound messages in `inputs` as a flat list """
        ref_compounds = self.find_messages(key='inputs', message_type=reaction_pb2.Compound, ref=True)
        act_compounds = self.find_messages(key='inputs', message_type=reaction_pb2.Compound, ref=False)
        report = diff_compound_list(ref_compounds, act_compounds)
        return report

    def evaluate_inputs_compounds_lol(self) -> CompoundLolDiffReport:
        """ compound messages in `inputs` as a list of list """
        ref_lol = [ri.components for ri in list(self.reaction_message_ref.inputs.values())]
        act_lol = [ri.components for ri in list(self.reaction_message.inputs.values())]
        report = diff_compound_lol(ref_lol, act_lol)
        return report

    def evaluate_conditions(self) -> ConditionsDiffReport:
        """ reaction conditions message """
        ref_conditions = self.reaction_message.conditions
        act_conditions = self.reaction_message_ref.conditions
        report = diff_conditions(ref_conditions, act_conditions)
        return report

    # TODO implement
    # def evaluate_outcomes(self):
    #     pass
    #
    # TODO implement
    # def evaluate_workups(self):
    #     pass


def evaluate_model_outputs(
        model_outputs: list[ModelOutput],
        report_compounds_list=True, report_compounds_lol=True, report_conditions=True
):
    records = []
    for model_output in tqdm(model_outputs):
        # record = model_output.as_dict()
        record = {"identifier": model_output.identifier}
        try:
            inference_evaluator = Evaluator(model_output)
            record["valid_ord"] = True
        except EvaluatorSniffError:
            record["valid_ord"] = False
            records.append(record)
            continue
        if report_compounds_list:
            record.update(
                {"inputs_compounds_list__" + k: v for k, v in
                 inference_evaluator.evaluate_inputs_compounds_list().dict().items() if k != 'index_match'}
            )
        if report_compounds_lol:
            record.update(
                {"inputs_compounds_lol__" + k: v for k, v in
                 inference_evaluator.evaluate_inputs_compounds_lol().dict().items()}
            )
        if report_conditions:
            record.update(
                {"conditions__" + k: v for k, v in
                 inference_evaluator.evaluate_conditions().dict().items()}
            )
        records.append(record)
    df = pd.DataFrame.from_records(records)
    return df
