import glob
import json
import os.path
from typing import TypedDict

from loguru import logger
from ord_schema.message_helpers import json_format
from ord_schema.message_helpers import load_message
from ord_schema.proto.dataset_pb2 import Dataset
from ord_schema.proto.reaction_pb2 import Reaction
from tqdm import tqdm

"""
filed name in the output of `protobuf.json_format` is by default camel case
use `json_format.MessageToJson(r, preserving_proto_field_name=True)` to output unmodified keys
"""


class LData(TypedDict):
    notes__procedureDetails: str  # for prompt
    conditions: dict
    reaction_id: str
    inputs: dict[str, dict]
    outcomes: list[dict]
    workups: list[dict]
    warning_messages: list[str]


class LDataExportError(Exception): pass


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


def reaction_to_llm_data(r: Reaction):
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


def convert_datasets(
        output_data_dir="data_from_pb_no_warning",
        local_data_folder="/home/qai/workplace/ord-data/data",
        uspto_only=True,
        keep_only_no_warning=True,
):
    dataset_files = sorted(glob.glob(f"{local_data_folder}/*/*.pb.gz"))
    logger.info(f"# of dataset files: {len(dataset_files)}")
    for f in tqdm(dataset_files):
        logger.info(f"convert dataset file: {f}")
        dataset = load_message(f, Dataset)
        if uspto_only and "uspto-grants-" not in dataset.name.lower():
            logger.critical(f"skipping non-USPTO dataset: {dataset.name}")
        output_file = f"{output_data_dir}/{dataset.dataset_id}.json"
        if os.path.isfile(output_file) and os.path.getsize(output_file) > 0:
            logger.info("output already exists, skipping conversion...")
            continue
        lds = []
        for r in dataset.reactions:
            try:
                ld = reaction_to_llm_data(r)
            except LDataExportError as e:
                logger.critical(f"excluding reaction: {r.reaction_id}\ndue to:{e}")
                continue
            if keep_only_no_warning and len(ld['warning_messages']) > 0:
                logger.critical(f"excluding reaction: {r.reaction_id}\ndue to warning msgs: {ld['warning_messages']}")
                continue
            lds.append(ld)
        with open(output_file, 'w') as output_fp:
            json.dump(lds, output_fp, indent=2)


if __name__ == '__main__':
    logger.remove(0)
    logger.add(__file__.replace(".py", ".log"))
    convert_datasets(keep_only_no_warning=True)
    # 190764 reactions 515 datasets
