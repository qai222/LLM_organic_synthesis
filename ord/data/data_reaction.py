from __future__ import annotations

from enum import Enum

from loguru import logger
from ord_schema.message_helpers import json_format
from ord_schema.proto import reaction_pb2
from pydantic import BaseModel

"""
used in exporting reaction data from pb
"""


class ReactionDataWarningType(str, Enum):
    COMPOUND_NAME_NO_FOUND = "COMPOUND_NAME_NO_FOUND"
    COMPOUND_INVALID_AMOUNT = "COMPOUND_INVALID_AMOUNT"
    PRODUCT_NO_FOUND = "PRODUCT_NO_FOUND"
    MISSING_IMPORTANT_KEY = "MISSING_IMPORTANT_KEY"


class ReactionDataWarning(BaseModel):
    type: ReactionDataWarningType
    message: str


class ReactionData(BaseModel):
    """ Data loaded from pb messages """

    reaction_id: str
    """ unique identifier of this reaction """

    procedure_text: str
    """ unstructured text describing a reaction """

    inputs: dict[str, dict] = dict()
    """ ord.Reaction.inputs field """

    conditions: dict = dict()
    """ ord.Reaction.conditions field """

    outcomes: list[dict] = []
    """ ord.Reaction.outcomes field """

    workups: list[dict] = []
    """ ord.Reaction.workups field """

    warning_messages: list[ReactionDataWarning] = []
    """ warnings from loading data """

    @staticmethod
    def clean_compound(compound: dict, message_prefix: str = "") -> list[ReactionDataWarning]:
        """
        clean up a compound message in-place

        :param compound:
        :return:
        """
        w_msgs = []

        # 1. remove non-name identifiers
        identifiers = compound['identifiers']
        new_identifiers = []
        for ii in identifiers:
            if ii['type'].upper() != 'NAME':
                continue
            else:
                new_identifiers.append(ii)
                break
        assert len(new_identifiers) <= 1
        if len(new_identifiers) == 0:
            warning = ReactionDataWarning(
                type=ReactionDataWarningType.COMPOUND_NAME_NO_FOUND,
                message=f"no name found in compound identifiers: {identifiers}"
            )
            w_msgs.append(warning)
        else:
            compound['identifiers'] = new_identifiers

        # 2. remove invalid amounts
        if 'amount' in compound:
            remove_amount = False
            for kk, vv in compound['amount'].items():
                if not (isinstance(vv, dict) and 'value' in vv):
                    continue
                if vv['value'] == 0:
                    warning = ReactionDataWarning(
                        type=ReactionDataWarningType.COMPOUND_INVALID_AMOUNT,
                        message=f"compound amount invalid, the amount will be removed: {kk}=={vv}"
                    )
                    w_msgs.append(warning)
                    remove_amount = True
                    break
            if remove_amount:
                compound.pop('amount', None)
        for msg in w_msgs:
            msg.message = message_prefix + msg.message
        return w_msgs

    @classmethod
    def from_reaction_message(cls, r: reaction_pb2.Reaction) -> ReactionData:

        logger.info(f"converting reaction: {r.reaction_id}")

        warning_messages = []

        rdict = json_format.MessageToDict(r, preserving_proto_field_name=True)
        # if `preserving_proto_field_name` is False, field name will automatically be converted to camel case

        try:
            procedure_text = rdict['notes']['procedure_details']
        except KeyError:
            raise ReactionDataError("no procedure text!")

        if "inputs" not in rdict:
            warning = ReactionDataWarning(
                type=ReactionDataWarningType.MISSING_IMPORTANT_KEY,
                message="missing key: ord.Reaction.inputs"
            )
            warning_messages.append(warning)
            reaction_inputs = dict()
        else:
            reaction_inputs = rdict["inputs"]
            for k, v in reaction_inputs.items():
                try:
                    components = v['components']
                except KeyError:
                    # possible if the input is represented by crude_components, ex. ord-7c920412f21b4b8195d3bf450f022cbd
                    # exclude them for now
                    raise ReactionDataError("cannot find components in reaction.inputs!")
                for ic, c in enumerate(components):
                    warning_messages += ReactionData.clean_compound(c, f"inputs.[{k}].[{ic}]: ")

        if "conditions" not in rdict:
            warning = ReactionDataWarning(
                type=ReactionDataWarningType.MISSING_IMPORTANT_KEY,
                message="missing key: ord.Reaction.conditions"
            )
            warning_messages.append(warning)
            reaction_conditions = {}
        else:
            reaction_conditions = rdict['conditions']
            reaction_conditions.pop("details", None)  # in USPTO this is just a pointer to `notes.procedure_details`

        if "outcomes" not in rdict:
            warning = ReactionDataWarning(
                type=ReactionDataWarningType.MISSING_IMPORTANT_KEY,
                message="missing key: ord.Reaction.outcomes"
            )
            warning_messages.append(warning)
            reaction_outcomes = []
        else:
            reaction_outcomes = rdict['outcomes']
            for i_outcome, outcome in enumerate(reaction_outcomes):
                try:
                    products = outcome['products']
                except KeyError:
                    warning = ReactionDataWarning(
                        type=ReactionDataWarningType.PRODUCT_NO_FOUND,
                        message=f"missing product in: ord.Reaction.outcomes.[{i_outcome}]"
                    )
                    warning_messages.append(warning)
                    products = []
                for ic, c in enumerate(products):
                    warning_messages += ReactionData.clean_compound(c, f"outcomes.products.[{ic}]")

        if "workups" not in rdict:
            warning = ReactionDataWarning(
                type=ReactionDataWarningType.MISSING_IMPORTANT_KEY,
                message="missing key: ord.Reaction.workups"
            )
            warning_messages.append(warning)
            reaction_workups = []
        else:
            reaction_workups = rdict["workups"]
            for w in reaction_workups:
                try:
                    workup_input = w['input']
                    components = workup_input['components']
                except KeyError:
                    continue
                for ic, c in enumerate(components):
                    warning_messages += ReactionData.clean_compound(c, f"workups.components.[{ic}]")

        ldict = ReactionData(
            procedure_text=procedure_text,
            conditions=reaction_conditions,
            reaction_id=rdict['reaction_id'],
            inputs=reaction_inputs,
            outcomes=reaction_outcomes,
            workups=reaction_workups,
            warning_messages=warning_messages,
        )
        return ldict


class ReactionDataError(Exception):
    pass
