from __future__ import annotations

import json
import os
import pathlib
import random
from typing import List

import matplotlib.pyplot as plt
import parse
import torch
from google.protobuf.json_format import ParseDict, MessageToDict
from loguru import logger
from matplotlib.patches import Rectangle
from ord_schema.proto.reaction_pb2 import Reaction, CompoundIdentifier, ProductMeasurement
from pandas._typing import FilePath
from pydantic import BaseModel
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from ord.data.data_reaction import ReactionData
from ord.utils import json_load, json_dump

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_PROMPT_TEMPLATE = "Below is a description of an organic reaction. Extract information from it to an ORD JSON record.\n\n### Procedure:\n{}\n\n### ORD JSON:\n"


def prompt_to_procedure(prompt: str) -> str:
    return parse.parse(DEFAULT_PROMPT_TEMPLATE, prompt)[0]


class Tokenizer:
    """
    # Copyright (c) Meta Platforms, Inc. and affiliates.
    # This software may be used and distributed according to the terms of the GNU General Public License version 3.
    """

    def __init__(self, model_path: FilePath):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)


class LlmData(BaseModel):
    reaction_id: str
    """ unique reaction identifier """

    reference_completion: str
    """ expected completion """

    prompt: str
    """ input prompt """

    n_token: int | None = None
    """ number of tokens for prompt + completion """

    @property
    def alpaca_dict(self) -> dict:
        return {
            "instruction": self.prompt,
            "output": self.reference_completion,
        }

    @property
    def example(self) -> str:
        return self.prompt + self.reference_completion

    @classmethod
    def from_reaction_data(cls, data: ReactionData, ord_fields: list[str], prompt_template: str, ) -> LlmData:
        """
        convert a `ReactionData` object to a `LlmData`

        :param data:
        :param ord_fields:
        :param prompt_template:
        :return:
        """
        reaction_id = data.reaction_id
        prompt = prompt_template.format(data.procedure_text)
        d = {}
        for field in ord_fields:
            value = getattr(data, field)
            if len(value):
                d[field] = value
        reference_completion = json.dumps(d)
        return cls(reaction_id=reaction_id, reference_completion=reference_completion, prompt=prompt)

    def get_n_token(self, tokenizer: Tokenizer) -> int:
        """
        calculate and set the n_token attribute

        :param tokenizer:
        :return:
        """
        example = torch.tensor(tokenizer.encode(self.example, bos=True, eos=True), dtype=torch.int64)
        self.n_token = example.shape[0]
        return self.n_token

    @staticmethod
    def get_identifiers_and_alpaca_dicts(data_list: list[LlmData]) -> tuple[list[str], list[dict]]:
        ids = []
        alpaca_dicts = []
        for d in data_list:
            ids.append(d.reaction_id)
            alpaca_dicts.append(d.alpaca_dict)
        return ids, alpaca_dicts


class LlmDataset(BaseModel):
    source_file: str | None = None
    """ source file if this dataset is created from another file """

    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    """ the template used to build prompts, must have a `procedure_text` placeholder """

    ord_root_fields: list[str] = []
    """ the fields that will be used to build prompts, a subset of {`inputs`, `conditions`, `workups`, `outcomes`} """

    tokenizer_path: str | None = None
    """ path to the tokenizer """

    data: list[LlmData] = []

    def load(self):
        assert os.path.isfile(self.source_file)
        tokenizer = Tokenizer(model_path=self.tokenizer_path)
        logger.info(f"loading source: {self.source_file}")
        reaction_data_list = json_load(self.source_file)
        logger.info("loaded!")
        for reaction_data in tqdm(reaction_data_list, desc="converting reaction data to LLM data"):
            reaction_data = ReactionData(**reaction_data)
            llm_data = LlmData.from_reaction_data(reaction_data, self.ord_root_fields, self.prompt_template)
            llm_data.get_n_token(tokenizer)
            self.data.append(llm_data)

    def plot_n_token_cdf(self, n_token_limit: int):
        n_accepted = 0
        n_total = len(self.data)
        n_tokens = []
        for d in self.data:
            n_tokens.append(d.n_token)
            if d.n_token <= n_token_limit:
                n_accepted += 1
        fig = plt.figure(figsize=(3.5, 3), layout="constrained")
        ax = fig.subplots(1, 1)
        line = ax.ecdf(n_tokens, c="k")
        line.remove()
        xs, ys = line.get_data()
        ax.scatter(xs, ys, c="k")
        ax.grid(True)
        ax.set_ylabel("Cumulative proportion of records")
        ax.set_xlabel("Number of tokens")
        ax.label_outer()
        rect = Rectangle((0, 0), n_token_limit, n_accepted / n_total, facecolor="k", alpha=0.5)
        accepted_percentage = "{:.2%}".format(n_accepted / n_total)
        logger.info(f"# limit/accepted/total: {n_token_limit}/{n_accepted}({accepted_percentage})/{n_total}")
        ax.add_patch(rect)
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim([0, min([max(n_tokens), 5000])])
        return fig

    def apply_token_limit(self, n_token_limit: int):
        """
        apply token limit, modify the data list in-place!

        :param n_token_limit:
        :return:
        """
        remove_indices = [i for i in range(len(self.data)) if self.data[i].n_token > n_token_limit]
        for i in sorted(remove_indices, reverse=True):
            del self.data[i]

    def split(
            self, total_size: int, n_token_limit: int, name: str, seed: int = 42,
            train_size: float = 0.8, valid_size: float = 0.1, test_size: float = 0.1,
            explicit_policy: int = 0,
    ):
        """
        make a train-valid-test split of the dataset

        :param explicit_policy: policy for fields not explicitly present in the procedure text
        :param n_token_limit: max token limit for the dataset
        :param total_size: total number of data points
        :param seed:
        :param train_size:
        :param valid_size:
        :param test_size:
        :return:
        """
        self.apply_token_limit(n_token_limit)

        assert total_size % 10 == 0
        assert total_size <= len(self.data)
        random.seed(seed)
        data = random.sample(self.data, k=total_size)
        random.shuffle(data)

        # this is probably not the best place to implement explicit policy...
        new_data = []
        if explicit_policy == 1:
            n_shrunk = 0
            data: list[LlmData]
            for d in data:
                procedure_text = prompt_to_procedure(d.prompt)
                ref_dict = json.loads(d.reference_completion)
                ref_reaction = ParseDict(ref_dict, Reaction())
                ref_reaction: Reaction
                for outcome in ref_reaction.outcomes:
                    for product in outcome.products:

                        # if product identifier not present explicitly, remove it
                        to_pop = []
                        for i, identifier in enumerate(product.identifiers):
                            identifier: CompoundIdentifier
                            if identifier.value.lower() not in procedure_text.lower():
                                to_pop.append(i)
                        to_pop = sorted(to_pop, reverse=True)
                        for i in to_pop:
                            product.identifiers.pop(i)

                        # if a yield value not present explicitly, remove it
                        to_pop = []
                        for i, measurement in enumerate(product.measurements):
                            measurement: ProductMeasurement
                            if measurement.percentage.value:  # if not a percentage measurement this returns zero
                                if measurement.percentage.value > 100 or str(
                                        int(measurement.percentage.value)) not in procedure_text.lower():
                                    to_pop.append(i)
                        to_pop = sorted(to_pop, reverse=True)
                        for i in to_pop:
                            product.measurements.pop(i)
                new_d = LlmData(
                    reaction_id=d.reaction_id,
                    reference_completion=json.dumps(MessageToDict(ref_reaction, preserving_proto_field_name=True)),
                    prompt=d.prompt,
                    n_token=None,
                )
                if len(new_d.reference_completion) < len(d.reference_completion):
                    n_shrunk += 1
                new_data.append(new_d)
            if not n_shrunk:
                logger.warning(
                    "no record was shrunk after applying explicit policy, this is abnormal if your dataset is large")
            data = new_data

        actual_train_size = round(total_size * train_size)
        actual_valid_size = round(total_size * valid_size)
        actual_test_size = round(total_size * test_size)
        assert actual_valid_size + actual_test_size + actual_train_size == total_size

        train_data = data[:actual_train_size]
        valid_data = data[actual_train_size:actual_train_size + actual_valid_size]
        test_data = data[-actual_test_size:]

        train_data_identifiers, train_data = LlmData.get_identifiers_and_alpaca_dicts(train_data)
        valid_data_identifiers, valid_data = LlmData.get_identifiers_and_alpaca_dicts(valid_data)
        test_data_identifiers, test_data = LlmData.get_identifiers_and_alpaca_dicts(test_data)

        assert not set(train_data_identifiers).intersection(set(test_data_identifiers))
        assert not set(valid_data_identifiers).intersection(set(test_data_identifiers))
        assert not set(train_data_identifiers).intersection(set(valid_data_identifiers))

        meta = {
            "n_token_limit": n_token_limit,
            "source_file": self.source_file,
            "prompt_template": self.prompt_template,
            "tokenizer_path": self.tokenizer_path,
            "ord_root_fields": self.ord_root_fields,
            "total_size": total_size,
            "train_size": train_size,
            "valid_size": valid_size,
            "test_size": test_size,
            "actual_train_size": actual_train_size,
            "actual_valid_size": actual_valid_size,
            "actual_test_size": actual_test_size,
            "train_data_identifiers": train_data_identifiers,
            "valid_data_identifiers": valid_data_identifiers,
            "test_data_identifiers": test_data_identifiers,
        }

        save_dir = name
        if explicit_policy:
            meta['explicit_policy'] = explicit_policy
            save_dir += f"_exp{explicit_policy}"
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

        json_dump(os.path.join(save_dir, f"meta.json"), meta, indent=2)
        json_dump(os.path.join(save_dir, f"train.json"), train_data, indent=2)
        json_dump(os.path.join(save_dir, f"valid.json"), valid_data, indent=2)
        json_dump(os.path.join(save_dir, f"test.json"), test_data, indent=2)
