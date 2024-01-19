from __future__ import annotations

import glob
import json
import os.path
import signal
from functools import wraps

import ord_schema.message_helpers
import pandas as pd
from google.protobuf import json_format
from loguru import logger
from ord_schema.proto import reaction_pb2
from pandas._typing import FilePath
from pydantic import BaseModel
from tqdm import tqdm

import ord_diff
from ord_diff.utils import json_load


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


def get_compounds(reaction_message, extracted_from: str):
    if extracted_from == "inputs":
        inputs = [*reaction_message.inputs.values()]
        mt = reaction_pb2.Compound
    elif extracted_from == "outcomes":
        inputs = [*reaction_message.outcomes]
        mt = reaction_pb2.ProductCompound
    elif extracted_from == "workups":
        inputs = [*reaction_message.workups]
        mt = reaction_pb2.Compound
    else:
        raise ValueError
    compounds = []
    for ri in inputs:
        compounds += ord_schema.message_helpers.find_submessages(ri, submessage_type=mt)
    return compounds


class PairEvaluator(BaseModel):
    parsed_model_output: ord_diff.ModelOutput

    reference_dict: dict

    inferred_dict: dict | None

    valid_json: bool

    valid_ord: bool

    @classmethod
    def load_from_inference_json(cls, infer_json: FilePath, prompt_template: str):

        prompt_header = prompt_template.split("\n")[0]
        response_header = prompt_template.split("\n")[-2]

        record = json_load(infer_json)
        ref_string = record['output']
        raw = record['response']
        model_output = ord_diff.ModelOutput.from_raw_alpaca(
            raw=raw,
            ref=ref_string,
            identifier=record['reaction_id'],
            prompt_template=prompt_template,
            prompt_header=prompt_header,
            response_header=response_header,
            instruction=record['instruction'],
        )

        # this round trip trick remove fields with empty values in dict
        ref_message = json_format.Parse(model_output.ref, reaction_pb2.Reaction())
        ref_dict = json_format.MessageToDict(ref_message)

        invalid_json = False
        invalid_ord = False

        try:
            json.loads(model_output.response)
        except json.decoder.JSONDecodeError as e:
            logger.error(e)
            invalid_json = True

        try:
            inferred_message = json_format.Parse(model_output.response, reaction_pb2.Reaction())
        except json_format.ParseError as e:
            logger.error(e)
            invalid_ord = True

        if invalid_json or invalid_ord:
            inferred_dict = None
        elif not invalid_ord:
            inferred_dict = json_format.MessageToDict(inferred_message)
        else:
            raise ValueError
        return cls(
            parsed_model_output=model_output,
            inferred_dict=inferred_dict,
            reference_dict=ref_dict,
            valid_ord=not invalid_ord,
            valid_json=not invalid_json,
        )

    @property
    def reaction_message_ref(self):
        return json_format.ParseDict(self.reference_dict, reaction_pb2.Reaction())

    @property
    def reaction_message_inf(self):
        if self.inferred_dict is None:
            raise ValueError
        return json_format.ParseDict(self.inferred_dict, reaction_pb2.Reaction())

    @property
    def identifier(self):
        return self.parsed_model_output.identifier

    @timeout(120)
    def eval_leaf_level_all(self):
        df1 = self.eval_leaf_level(message_type=ord_diff.MessageType.COMPOUND, extracted_from="inputs")
        df2 = self.eval_leaf_level(message_type=ord_diff.MessageType.PRODUCT_COMPOUND, extracted_from="outcomes")
        df3 = self.eval_leaf_level(message_type=ord_diff.MessageType.REACTION_CONDITIONS, extracted_from="")
        df4 = self.eval_leaf_level(message_type=ord_diff.MessageType.REACTION_WORKUP, extracted_from="")
        return pd.concat([df1, df2, df3, df4], axis=0)

    @timeout(120)
    def eval_message_level_all(self):
        data = dict(reaction_id=self.identifier)
        data.update(self.eval_message_level(message_type=ord_diff.MessageType.COMPOUND, extracted_from="inputs",
                                            return_record_with_header=True))
        data.update(
            self.eval_message_level(message_type=ord_diff.MessageType.PRODUCT_COMPOUND, extracted_from="outcomes",
                                    return_record_with_header=True))
        data.update(self.eval_message_level(message_type=ord_diff.MessageType.REACTION_CONDITIONS, extracted_from="",
                                            return_record_with_header=True))
        data.update(self.eval_message_level(message_type=ord_diff.MessageType.REACTION_WORKUP, extracted_from="",
                                            return_record_with_header=True))
        return data

    def eval_message_level(self, message_type: ord_diff.MessageType, extracted_from: str,
                           return_record_with_header: bool = True):

        text = self.parsed_model_output.instruction

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
        if message_type == ord_diff.MessageType.REACTION_CONDITIONS:
            conditions_inf = self.reaction_message_inf.conditions
            conditions_ref = self.reaction_message_ref.conditions
            assert conditions_ref is not None
            assert conditions_inf is not None
            mt = ord_diff.MessageType.REACTION_CONDITIONS
            diff = ord_diff.MDictDiff.from_message_pair(conditions_ref, conditions_inf, mt, text, text)
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
        elif message_type in (ord_diff.MessageType.COMPOUND, ord_diff.MessageType.PRODUCT_COMPOUND):
            messages_inf = get_compounds(self.reaction_message_inf, extracted_from=extracted_from)
            messages_ref = get_compounds(self.reaction_message_ref, extracted_from=extracted_from)
        elif message_type == ord_diff.MessageType.REACTION_WORKUP:
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
            diff = ord_diff.MDictListDiff.from_message_list_pair(
                messages_ref, messages_inf, message_type, text, text
            )
            record["n_removal"] = diff.n_absent
            record["n_addition"] = diff.n_excess
            record["n_alteration_strict"] = diff.n_changed_strict
            record["n_alteration_nonstrict"] = diff.n_changed_nonstrict
            record["n_intact_strict"] = diff.n_md1 - diff.n_changed_strict - diff.n_absent
            record["n_intact_nonstrict"] = diff.n_md1 - diff.n_changed_nonstrict - diff.n_absent
        if return_record_with_header:
            record = {message_type.value + "__" + k: v for k, v in record.items()}
        return record

    def eval_leaf_level(self, message_type: ord_diff.MessageType, extracted_from: str) -> pd.DataFrame:

        text = self.parsed_model_output.instruction

        # special treatment for conditions as it is a message rather than a list of message
        if message_type == ord_diff.MessageType.REACTION_CONDITIONS:
            conditions_inf = self.reaction_message_inf.conditions
            conditions_ref = self.reaction_message_ref.conditions
            assert conditions_ref is not None
            assert conditions_inf is not None
            mt = ord_diff.MessageType.REACTION_CONDITIONS
            diff = ord_diff.MDictDiff.from_message_pair(conditions_ref, conditions_inf, mt, text, text)
            df = ord_diff.report_diff(diff, message_type)
            df["reaction_id"] = self.identifier
            return df

        elif message_type in (ord_diff.MessageType.COMPOUND, ord_diff.MessageType.PRODUCT_COMPOUND):
            messages_inf = get_compounds(self.reaction_message_inf, extracted_from=extracted_from)
            messages_ref = get_compounds(self.reaction_message_ref, extracted_from=extracted_from)
        elif message_type == ord_diff.MessageType.REACTION_WORKUP:
            messages_inf = self.reaction_message_inf.workups
            messages_ref = self.reaction_message_ref.workups
        else:
            raise ValueError

        if len(messages_ref) == 0:
            dfs = []
            for m in messages_inf:
                md = ord_diff.MDict.from_message(m, message_type, text)
                df = ord_diff.report_diff_leafs(md, ct=ord_diff.DeltaType.ADDITION, from_m1=False)
                dfs.append(df)
            if len(dfs) == 0:
                df = pd.DataFrame()
            else:
                df = pd.concat(dfs, axis=0)
        elif len(messages_inf) == 0:
            dfs = []
            for m in messages_ref:
                md = ord_diff.MDict.from_message(m, message_type, text)
                df = ord_diff.report_diff_leafs(md, ct=ord_diff.DeltaType.REMOVAL, from_m1=True)
                dfs.append(df)
            if len(dfs) == 0:
                df = pd.DataFrame()
            else:
                df = pd.concat(dfs, axis=0)
        else:
            diff = ord_diff.MDictListDiff.from_message_list_pair(
                messages_ref, messages_inf, message_type, text, text
            )
            df = ord_diff.report_diff_list(diff, message_type=message_type)
        df["reaction_id"] = self.identifier
        return df

class BatchEvaluator(BaseModel):
    """ evaluate inferences produced by a finetuned model """

    inference_folder: str
    """ a folder contains `ord-*.json` files """

    data_folder: str
    """ where is the data used to train this model? """

    pair_evaluators: list[PairEvaluator]

    def get_pe(self, reaction_id: str) -> PairEvaluator:
        pe_dict = {pe.identifier: pe for pe in self.pair_evaluators}
        return pe_dict[reaction_id]

    def __getitem__(self, item: int):
        return self.pair_evaluators[item]

    def eval_leaf_level(self) -> pd.DataFrame:
        dfs = []
        for pe in tqdm(self.pair_evaluators, desc="batch evaluation -- leaf level"):
            if pe.valid_ord:
                try:
                    df = pe.eval_leaf_level_all()
                except:
                    logger.critical(f"evaluation error for: {pe.identifier}")
                    continue
                if df is None:
                    logger.critical(f"timeout error for: {pe.identifier}")
                    continue
                dfs.append(df)
        return pd.concat(dfs, axis=0)


    def eval_message_level(self) -> pd.DataFrame:
        records = []
        n_invalid_json = 0
        n_invalid_ord = 0
        for pe in tqdm(self.pair_evaluators, desc="batch evaluation -- message level"):
            if pe.valid_ord:
                try:
                    record = pe.eval_message_level_all()
                except:
                    logger.critical(f"evaluation error for: {pe.identifier}")
                    continue
                if record is None:
                    logger.critical(f"timeout error for: {pe.identifier}")
                    continue
                records.append(record)
            if not pe.valid_ord:
                n_invalid_ord += 1
            if not pe.valid_json:
                n_invalid_json += 1
        logger.critical(f"total/invalid JSON/invalid ORD: {len(self.pair_evaluators)}/{n_invalid_json}/{n_invalid_ord}")
        return pd.DataFrame.from_records(records)

    @classmethod
    def load_inferences(cls, inference_folder: FilePath, data_folder: FilePath, first_k: int = None):
        """
        load inferences from model outputs

        :param inference_folder: location of model outputs
        :param data_folder: location of training data (only need prompt info from there)
        :param first_k: if only load the first k inferences
        :return:
        """
        json_files = sorted(glob.glob(f"{inference_folder}/ord*.json"))
        if first_k:
            json_files = json_files[:first_k]

        logger.warning(f"this batch of inferences was generated using test data set from: {data_folder}")
        prompt_template = json_load(os.path.join(data_folder, "params.json"))['prompt_template']
        logger.warning(f"prompt template is:\n{prompt_template}")

        # prompt_header = prompt_template.split("\n")[0]
        # response_header = prompt_template.split("\n")[-2]
        # logger.warning(f"prompt header is: {prompt_header}")
        # logger.warning(f"response header is: {response_header}")
        pes = []
        for infer_json in json_files:
            # for infer_json in tqdm(json_files, desc="loading infer json files"):
            pe = PairEvaluator.load_from_inference_json(infer_json, prompt_template)
            pes.append(pe)
        return cls(inference_folder=inference_folder, data_folder=data_folder, pair_evaluators=pes, )
