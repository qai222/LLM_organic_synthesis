from __future__ import annotations

from enum import Enum
from typing import TypedDict

from loguru import logger
from ord_schema.message_helpers import json_format
from ord_schema.proto.reaction_pb2 import Reaction

"""
used in exporting reaction data ready for LLM
"""

class LData(TypedDict):
    reaction_id: str
    notes__procedureDetails: str  # for prompt
    conditions: dict
    inputs: dict[str, dict]
    outcomes: list[dict]
    workups: list[dict]
    warning_messages: list[str]


class LDataExportError(Exception):
    pass


class CreLabel(str, Enum):
    O = "O"
    B_Reactants = "B-Reactants"
    I_Reactants = "I-Reactants"
    B_Catalyst_Reagents = "B-Catalyst_Reagents"
    I_Catalyst_Reagents = "I-Catalyst_Reagents"
    B_Workup_reagents = "B-Workup_reagents"
    I_Workup_reagents = "I-Workup_reagents"
    B_Reaction = "B-Reaction"
    I_Reaction = "I-Reaction"
    B_Solvent = "B-Solvent"
    I_Solvent = "I-Solvent"
    B_Yield = "B-Yield"
    I_Yield = "I-Yield"
    B_Temperature = "B-Temperature"
    I_Temperature = "I-Temperature"
    B_Time = "B-Time"
    I_Time = "I-Time"
    B_Prod = "B-Prod"
    I_Prod = "I-Prod"

    @property
    def role(self):
        if self == CreLabel.O:
            return self
        return self[2:]

    @property
    def type(self):
        if self == CreLabel.O:
            return self
        return self[0]


class CreReaction(TypedDict):
    """ from https://github.com/jiangfeng1124/ChemRxnExtractor/blob/main/configs/role_labels.txt """

    reaction_index: int  # the same sentence can describe multiple reactions, this is the column index in role.txt

    passage_id: str
    sentence_id: int

    Reactants: list[str]
    Catalyst_Reagents: list[str]
    Workup_reagents: list[str]
    Prod: str
    Reaction: str
    Solvent: str
    Yield: str
    Temperature: str
    Time: str


class CrePassage(TypedDict):
    passage_id: str
    passage_text: str
    is_subpassage: bool  # is this constructed from role.txt (True) or prod.txt (False)?
    contains: list[CreReaction]  # reaction indices


def clean_compound(compound: dict, msg_prefix: str = "") -> list[str]:
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
        msg = f"no name found in compound identifiers: {identifiers}"
        w_msgs.append(msg)
    else:
        compound['identifiers'] = new_identifiers

    # 2. remove invalid amounts
    if 'amount' in compound:
        remove_amount = False
        for kk, vv in compound['amount'].items():
            if not (isinstance(vv, dict) and 'value' in vv):
                continue
            if vv['value'] == 0:
                msg = f"compound amount invalid, the amount will be removed: {compound}"
                # raise LDataExportError(msg)
                w_msgs.append(msg)
                remove_amount = True
                break
        if remove_amount:
            compound.pop('amount', None)
    return [msg_prefix + ": " + m for m in w_msgs]


def reaction_to_llm_data(r: Reaction) -> LData:
    logger.info(f"converting reaction: {r.reaction_id}")
    w_msgs = []

    rdict = json_format.MessageToDict(r)

    r_key = 'inputs'
    if r_key not in rdict:
        msg = f"r_key missing: {r_key}"
        w_msgs.append(msg)
        r_inputs = dict()
    else:
        r_inputs = rdict[r_key]
        for k, v in r_inputs.items():
            try:
                components = v['components']
            except KeyError:
                # possible if the input is represented by crude_components, ex. ord-7c920412f21b4b8195d3bf450f022cbd
                # exclude them for now
                raise LDataExportError("cannot find components in reaction.inputs!")
            for ic, c in enumerate(components):
                w_msgs += clean_compound(c, f"inputs.[{k}].[{ic}]")

    r_key = 'workups'
    if r_key not in rdict:
        # msg = f"r_key missing: {r_key}"  # this can be absent
        # w_msgs.append(msg)
        r_workups = []
    else:
        r_workups = rdict[r_key]
        for w in r_workups:
            try:
                workup_input = w['input']
                components = workup_input['components']
            except KeyError:
                continue
            for ic, c in enumerate(components):
                w_msgs += clean_compound(c, f"workups.components.[{ic}]")

    r_key = 'outcomes'
    if r_key not in rdict:
        msg = f"r_key missing: {r_key}"
        w_msgs.append(msg)
        r_outcomes = []
    else:
        r_outcomes = rdict[r_key]
        for outcome in r_outcomes:
            try:
                products = outcome['products']
            except KeyError:
                raise LDataExportError("no products in outcome!")
            for ic, c in enumerate(products):
                w_msgs += clean_compound(c, f"outcomes.products.[{ic}]")

    try:
        prompt = rdict['notes']['procedureDetails']
    except KeyError:
        raise LDataExportError("no input text!")

    if "conditions" not in rdict:
        rdict['conditions'] = {}

    ldict = LData(
        notes__procedureDetails=prompt,
        conditions=rdict['conditions'],
        reaction_id=rdict['reactionId'],
        inputs=r_inputs,
        outcomes=r_outcomes,
        workups=r_workups,
        warning_messages=w_msgs,
    )
    return ldict
