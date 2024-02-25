from __future__ import annotations

import glob
import json
import os.path

import pandas as pd
from google.protobuf import json_format
from json_repair import repair_json
from loguru import logger
from ord_schema.proto import reaction_pb2
from pandas._typing import FilePath
from pydantic import BaseModel

from ..diff.base import MessageType, DeltaType
from ..diff.report import report_diff, report_diff_leafs, report_diff_list
from ..diff.schema import MDictDiff, MDictListDiff, MDict
from ..utils import json_load, get_compounds, timeout


class PairEvaluator(BaseModel):
    """ a pair of reaction messages in python dictionary form """

    reaction_id: str
    """ the identifier maps to a unique procedure text """

    reference_text: str
    """ the original JSON string """

    inference_text: str
    """ the inferred string """

    valid_json: bool
    """ if this is a valid json """

    valid_ord: bool
    """ if this is a valid ord json """

    @classmethod
    def from_texts(cls, ref_text: str, inf_text: str, reaction_id: str) -> PairEvaluator:
        invalid_json = False
        invalid_ord = False

        try:
            json.loads(inf_text)
        except json.decoder.JSONDecodeError as e:
            logger.error(e)
            invalid_json = True

        try:
            json_format.Parse(inf_text, reaction_pb2.Reaction())
        except json_format.ParseError as e:
            logger.error(e)
            invalid_ord = True

        return cls(
            reaction_id=reaction_id,
            reference_text=ref_text,
            inference_text=inf_text,
            valid_json=not invalid_json,
            valid_ord=not invalid_ord,

        )

    def repair(self) -> PairEvaluator:
        """ use `json_repair` to repair the output https://github.com/mangiucugna/json_repair/ """
        assert not self.valid_json, "you are repairing a valid JSON!"
        repaired_string = repair_json(self.inference_text)
        return PairEvaluator.from_texts(self.reference_text, repaired_string, self.reaction_id)

    @classmethod
    def from_inference_folder_cot(cls, inference_folder_cot: FilePath, dataset_json: FilePath) -> list[PairEvaluator]:
        cot_responses = dict()
        for jf in sorted(glob.glob(f"{inference_folder_cot}/ord-*.json")):
            rid = os.path.basename(jf).replace(".json", "")
            ord_json_string = json_load(jf)['choices'][0]['message']['content']
            if "###" in ord_json_string:
                ord_json_string = ord_json_string.split("###")[1]
            cot_responses[rid] = ord_json_string
        data = json_load(dataset_json)
        cot_reference_data = {d["reaction_id"]: d for d in data}
        pes = []
        for rid in cot_responses:
            ref_text = cot_reference_data[rid]['reference_string']
            inf_text = cot_responses[rid]
            pe = PairEvaluator.from_texts(ref_text=ref_text, inf_text=inf_text, reaction_id=rid)
            pes.append(pe)
        return pes

    @classmethod
    def from_inference_folder(cls, inference_folder: FilePath, dataset_folder: FilePath) -> list[PairEvaluator]:
        inferred_data = []
        for json_file in sorted(glob.glob(f"{inference_folder}/*.json")):
            inferred_data += json_load(json_file)

        if isinstance(inferred_data, list):
            inf_strings = inferred_data
        elif isinstance(inferred_data, dict):
            inf_strings = [inferred_data, ]
        else:
            raise TypeError

        test_dataset = json_load(f"{dataset_folder}/test.json")
        meta_info = json_load(f"{dataset_folder}/meta.json")
        test_dataset_ids = meta_info['test_data_identifiers']
        assert len(test_dataset_ids) == len(
            inferred_data), f"test set has # {len(test_dataset_ids)}, inferred has # {len(inferred_data)}"
        pes = []
        for inf_string, test_record, test_id in zip(inf_strings, test_dataset, test_dataset_ids):
            ref_string = test_record['output']
            pes.append(PairEvaluator.from_texts(ref_text=ref_string, inf_text=inf_string, reaction_id=test_id))
        return pes

    @property
    def reaction_message_ref(self) -> reaction_pb2.Reaction:
        return json_format.Parse(self.reference_text, reaction_pb2.Reaction())

    @property
    def reaction_message_inf(self) -> reaction_pb2.Reaction:
        return json_format.Parse(self.inference_text, reaction_pb2.Reaction())

    def eval_message_level(
            self, message_type: MessageType, extracted_from: str,
            return_record_with_header: bool = True
    ) -> dict:

        record = {
            "n_ref": 0,
            "n_inf": 0,
            "n_removal": 0,
            "n_addition": 0,
            "n_alteration_strict": 0,
            "n_alteration_nonstrict": 0,
            "n_intact_strict": 0,
            "n_intact_nonstrict": 0,
        }

        # special treatment for conditions as it is a message rather than a list of message
        if message_type == MessageType.REACTION_CONDITIONS:
            conditions_inf = self.reaction_message_inf.conditions
            conditions_ref = self.reaction_message_ref.conditions
            assert conditions_ref is not None
            assert conditions_inf is not None
            mt = MessageType.REACTION_CONDITIONS
            diff = MDictDiff.from_message_pair(conditions_ref, conditions_inf, mt)
            record["n_ref"] = 1
            record["n_inf"] = 1
            record["n_removal"] = 0
            record["n_addition"] = 0
            record["n_alteration_strict"] = 1 if diff.is_altered(strict=True) else 0
            record["n_alteration_nonstrict"] = 1 if diff.is_altered(strict=False) else 0
            record["n_intact_strict"] = 0 if diff.is_altered(strict=True) else 1
            record["n_intact_nonstrict"] = 0 if diff.is_altered(strict=False) else 1
            if return_record_with_header:
                record = {message_type.value + "__" + k: v for k, v in record.items()}
            return record
        elif message_type in (MessageType.COMPOUND, MessageType.PRODUCT_COMPOUND):
            messages_inf = get_compounds(self.reaction_message_inf, extracted_from=extracted_from)
            messages_ref = get_compounds(self.reaction_message_ref, extracted_from=extracted_from)
        elif message_type == MessageType.REACTION_WORKUP:
            messages_inf = self.reaction_message_inf.workups
            messages_ref = self.reaction_message_ref.workups
        else:
            raise ValueError
        record["n_ref"] = len(messages_ref)
        record["n_inf"] = len(messages_inf)
        if len(messages_ref) == 0 or len(messages_inf) == 0:
            record["n_removal"] = len(messages_ref)
            record["n_addition"] = len(messages_inf)
            record["n_alteration_strict"] = 0
            record["n_alteration_nonstrict"] = 0
            record["n_intact_strict"] = 0
            record["n_intact_nonstrict"] = 0
        else:
            diff = MDictListDiff.from_message_list_pair(messages_ref, messages_inf, message_type)
            record["n_removal"] = diff.n_absent
            record["n_addition"] = diff.n_excess
            record["n_alteration_strict"] = diff.n_changed_strict
            record["n_alteration_nonstrict"] = diff.n_changed_nonstrict
            record["n_intact_strict"] = diff.n_md1 - diff.n_changed_strict - diff.n_absent
            record["n_intact_nonstrict"] = diff.n_md1 - diff.n_changed_nonstrict - diff.n_absent
        if return_record_with_header:
            record = {message_type.value + "__" + k: v for k, v in record.items()}
        return record

    def eval_leaf_level(self, message_type: MessageType, extracted_from: str) -> pd.DataFrame:

        # special treatment for conditions as it is a message rather than a list of message
        if message_type == MessageType.REACTION_CONDITIONS:
            conditions_inf = self.reaction_message_inf.conditions
            conditions_ref = self.reaction_message_ref.conditions
            assert conditions_ref is not None
            assert conditions_inf is not None
            mt = MessageType.REACTION_CONDITIONS
            diff = MDictDiff.from_message_pair(conditions_ref, conditions_inf, mt)
            df = report_diff(diff, message_type)
            df["reaction_id"] = self.reaction_id
            return df

        elif message_type in (MessageType.COMPOUND, MessageType.PRODUCT_COMPOUND):
            messages_inf = get_compounds(self.reaction_message_inf, extracted_from=extracted_from)
            messages_ref = get_compounds(self.reaction_message_ref, extracted_from=extracted_from)
        elif message_type == MessageType.REACTION_WORKUP:
            messages_inf = self.reaction_message_inf.workups
            messages_ref = self.reaction_message_ref.workups
        else:
            raise ValueError

        if len(messages_ref) == 0:
            dfs = []
            for m in messages_inf:
                md = MDict.from_message(m, message_type)
                df = report_diff_leafs(md, ct=DeltaType.ADDITION, from_m1=False)
                dfs.append(df)
            if len(dfs) == 0:
                df = pd.DataFrame()
            else:
                df = pd.concat(dfs, axis=0)
        elif len(messages_inf) == 0:
            dfs = []
            for m in messages_ref:
                md = MDict.from_message(m, message_type)
                df = report_diff_leafs(md, ct=DeltaType.REMOVAL, from_m1=True)
                dfs.append(df)
            if len(dfs) == 0:
                df = pd.DataFrame()
            else:
                df = pd.concat(dfs, axis=0)
        else:
            diff = MDictListDiff.from_message_list_pair(messages_ref, messages_inf, message_type)
            df = report_diff_list(diff, message_type=message_type)
        df["reaction_id"] = self.reaction_id
        return df

    @timeout(30)
    def eval_leaf_level_all(self) -> pd.DataFrame:
        dfs = []
        for message_type, extracted_from in (
                (MessageType.COMPOUND, "inputs"),
                (MessageType.PRODUCT_COMPOUND, "outcomes"),
                (MessageType.REACTION_CONDITIONS, ""),
                (MessageType.REACTION_WORKUP, ""),
        ):
            df = self.eval_leaf_level(message_type=message_type, extracted_from=extracted_from)
            dfs.append(df)
        return pd.concat(dfs, axis=0)

    @timeout(30)
    def eval_message_level_all(self) -> dict:
        data = dict(reaction_id=self.reaction_id)
        for message_type, extracted_from in (
                (MessageType.COMPOUND, "inputs"),
                (MessageType.PRODUCT_COMPOUND, "outcomes"),
                (MessageType.REACTION_CONDITIONS, ""),
                (MessageType.REACTION_WORKUP, ""),
        ):
            data.update(
                self.eval_message_level(
                    message_type=message_type, extracted_from=extracted_from, return_record_with_header=True
                )
            )
        return data
