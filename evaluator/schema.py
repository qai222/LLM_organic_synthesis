from __future__ import annotations

import json
from typing import Literal

from ord_schema.message_helpers import json_format, find_submessages, Type, MessageType
from ord_schema.proto import reaction_pb2

from evaluator.ord_dd import diff_compound_list, diff_conditions, diff_compound_lol


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

    def evaluate_inputs_compounds_list(self):
        """ compound messages in `inputs` as a flat list """
        ref_compounds = self.find_messages(key='inputs', message_type=reaction_pb2.Compound, ref=True)
        act_compounds = self.find_messages(key='inputs', message_type=reaction_pb2.Compound, ref=False)
        report = diff_compound_list(ref_compounds, act_compounds)
        return report

    def evaluate_inputs_compounds_lol(self):
        """ compound messages in `inputs` as a list of list """
        ref_lol = [ri.components for ri in
                   self.find_messages(key='inputs', message_type=reaction_pb2.ReactionInput, ref=True)]
        act_lol = [ri.components for ri in
                   self.find_messages(key='inputs', message_type=reaction_pb2.ReactionInput, ref=False)]
        report = diff_compound_lol(ref_lol, act_lol)
        return report

    def evaluate_conditions(self):
        """ reaction conditions message """
        ref_conditions = self.reaction_message.conditions
        act_conditions = self.reaction_message_ref.conditions
        report = diff_conditions(ref_conditions, act_conditions)
        return report

    def evaluate_outcomes(self):
        # TODO implement
        pass

    def evaluate_workups(self):
        # TODO implement
        pass
