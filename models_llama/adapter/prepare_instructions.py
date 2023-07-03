from __future__ import annotations

import glob
import json
import os
import random
from typing import Any
from typing import List

import torch
from loguru import logger
from pydantic import BaseModel
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


# TODO figure out the context window of llama and if it can change after fine tuning


class Tokenizer:
    """
    # Copyright (c) Meta Platforms, Inc. and affiliates.
    # This software may be used and distributed according to the terms of the GNU General Public License version 3.
    """

    def __init__(self, model_path: str = "7B_tokenizer/tokenizer.model"):
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


class InstructionDataset(BaseModel):
    ord_dataset_files: list[str] = sorted(
        glob.glob(f"{THIS_DIR}/../../data_from_pb_no_warning_20230416/data_from_pb_no_warning/*.json"))

    ord_procedure_field: str = "notes__procedureDetails"

    ord_target_fields: list[str] = ["inputs", ]

    prompt_template: str = "### Procedure:\n{instruction}\n\n### ORD-JSON:\n"

    tokenizer_path: str = f"{THIS_DIR}/7B_tokenizer/tokenizer.model"

    max_token: int = 900  # fits 24 gb gpu

    train_size: int = 10000

    test_size: int = 2000

    seed: int = 42

    train_data: list[dict] = []

    test_data: list[dict] = []

    all_alpaca_dicts: list[dict] = []

    def collect_reactions(self) -> list[dict]:
        reactions = []
        for jp in tqdm(self.ord_dataset_files):
            with open(jp, "r") as f:
                batch_reactions = json.load(f)
                reactions += batch_reactions
        return reactions

    def reaction_dict_to_alpaca_dict(self, r: dict[str, Any]):
        output = {k: r[k] for k in self.ord_target_fields}
        # TODO this should be done in extracting from postgres...
        if 'conditions' in self.ord_target_fields:
            output['conditions'].pop('details', None)
        if 'workups' in self.ord_target_fields:
            for w in output['workups']:
                w.pop('details')
        if 'outcomes' in self.ord_target_fields:
            for reaction_outcome in output['outcomes']:
                for analysis in reaction_outcome['analyses'].values():
                    analysis.pop('details')
        d = {
            "reaction_id": r['reaction_id'],
            "instruction": r[self.ord_procedure_field],
            "output": json.dumps(output),
        }

        for v in d.values():
            assert isinstance(v, str)
        return d

    @staticmethod
    def get_n_tokens(example: str, tokenizer: Tokenizer):
        example = torch.tensor(tokenizer.encode(example, bos=True, eos=False), dtype=torch.int64)
        return example.shape[0]

    def save(self):
        if len(self.train_data) == 0:
            self.load()
        filename = f"OrdAlpaca_MaxToken{self.max_token}_TrainSize{self.train_size}_TestSize{self.test_size}_{'-'.join(self.ord_target_fields)}.json"
        ds = self.json()
        with open(filename, "w") as f:
            f.write(ds)

    def load(self):
        tmp_filename = f"prepare_instructions_alpaca_dicts_tokenized_{'-'.join(self.ord_target_fields)}.json"
        try:
            with open(tmp_filename, "r") as f:
                all_alpaca_dicts = json.load(f)
        except (FileNotFoundError, ValueError) as e:
            tokenizer = Tokenizer(model_path=self.tokenizer_path)
            all_reactions = self.collect_reactions()
            len_chars = []
            len_tokens = []
            all_alpaca_dicts = []
            for r in tqdm(all_reactions):
                d = self.reaction_dict_to_alpaca_dict(r)
                all_alpaca_dicts.append(d)
                example = self.prompt_template.format_map({"instruction": d['instruction']}) + d['output']
                n_token = self.get_n_tokens(example, tokenizer)
                d['n_token_example'] = n_token
                len_chars.append(len(example))
                len_tokens.append(n_token)
            with open(tmp_filename, "w") as f:
                json.dump(all_alpaca_dicts, f)

        self.all_alpaca_dicts = all_alpaca_dicts

        accepted_alpaca_dicts = []
        for d in all_alpaca_dicts:
            n_token = d['n_token_example']
            if n_token <= self.max_token:
                accepted_alpaca_dicts.append(d)
        logger.info(f"# of all alpaca dicts: {len(all_alpaca_dicts)}")
        logger.info(
            f"# of accepted alpaca dicts: {len(accepted_alpaca_dicts)} ({round(len(accepted_alpaca_dicts) / len(all_alpaca_dicts), 4) * 100}%)")

        random.seed(self.seed)
        accepted_alpaca_dicts = random.sample(accepted_alpaca_dicts, self.test_size + self.train_size)
        random.shuffle(accepted_alpaca_dicts)
        train, test = accepted_alpaca_dicts[:self.train_size], accepted_alpaca_dicts[self.train_size:]
        self.train_data = train
        self.test_data = test

    def plot_ecdf_n_token_example(self):
        assert len(self.train_data) > 0
        import plotly.express as px
        import plotly
        fig = px.ecdf(
            x=[d['n_token_example'] for d in self.all_alpaca_dicts],
            marginal="histogram",
            labels={
                "x": "n_token(prompt + response)",
            }
        )
        plotly.offline.plot(fig, filename=f'ecdf_n_token_{"-".join(self.ord_target_fields)}.html')


if __name__ == '__main__':
    dataset = InstructionDataset(
        ord_target_fields=[
            'inputs',
            'conditions',
            'outcomes',
            'workups',
        ]
    )
    dataset.save()
    dataset.plot_ecdf_n_token_example()
