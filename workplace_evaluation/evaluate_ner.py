from __future__ import annotations

import os.path
import pathlib
import pprint

from Levenshtein import distance as levenshtein_distance
from loguru import logger
from ord_schema.message_helpers import get_compound_name
from ord_schema.proto import reaction_pb2
from pydantic import BaseModel

from ord.evaluation.evaluate_functions import get_pair_evaluators
from ord.utils import json_load, get_compounds

_THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))


class FuzzyToken(BaseModel):
    actual_token: str
    """ the actual string from dataset or inferences"""

    mapped_tokens: list[str]
    """ the token (from splitting text) it mapped to """

    mapped_token_positions: list[int]
    """ the indices of mapped tokens """

    role: int | None = None

    @classmethod
    def from_actual_token(cls, actual_token: str, text: str, role: int = None):
        _actual_token = actual_token
        if " " in actual_token:
            actual_token = sorted(actual_token.split(), key=lambda x: len(x))[-1]

        mts = []
        mtps = []
        for i, t in enumerate(text.split()):
            hit = False
            t_ = t.lower().replace("-", "")
            actual_token_ = actual_token.lower().replace("-", "")
            if t == actual_token:
                hit = True
            elif t_ == actual_token_:
                hit = True
                logger.critical(f"'{t}' from ref is matched to '{actual_token}' from inf")
            elif (len(t_) >= 5 and levenshtein_distance(t_, actual_token_) / len(actual_token_) <= 0.2 and
                  (actual_token_ in t_ or t_ in actual_token_)):
                hit = True
                logger.critical(f"token from text has matching:\n ('{t}', '{actual_token}')\noriginal text: {text}")
            if hit:
                mts.append(t)
                mtps.append(i)
        return cls(actual_token=_actual_token, mapped_tokens=mts, mapped_token_positions=mtps, role=role)

    def __eq__(self, other: FuzzyToken):
        eq_token = False
        eq_role = False

        if self.role is None and other.role is None:
            eq_role = True
        elif self.role == other.role:
            eq_role = True

        if (self.actual_token.lower().replace("-", "") ==
                other.actual_token.lower().replace("-", "")):
            eq_token = True
        distances = []
        for i in self.mapped_token_positions:
            for j in other.mapped_token_positions:
                distances.append(abs(i - j))
        if len(distances) == 0:
            eq_token = False
        elif min(distances) <= 2:
            eq_token = True
        return eq_token and eq_role

    @staticmethod
    def list_intersection(l1: list[FuzzyToken], l2: list[FuzzyToken]):
        return list(zip([ft1 for ft1 in l1 if ft1 in l2], [ft2 for ft2 in l2 if ft2 in l1]))

    @staticmethod
    def list_difference(l1: list[FuzzyToken], l2: list[FuzzyToken]):
        return [ft for ft in l1 if ft not in l2]


def get_ne(c: reaction_pb2.Compound | reaction_pb2.ProductCompound, include_role: bool):
    if include_role:
        return get_compound_name(c), c.reaction_role
    else:
        return get_compound_name(c)


def count_found(ref_tokens: list[str], inf_tokens: list[str], text: str, include_role=True, fuzz=True):
    found = sorted(set(ref_tokens).intersection(set(inf_tokens)))
    not_found_in_ref = sorted(set(ref_tokens).difference(found))
    not_found_in_inf = sorted(set(inf_tokens).difference(found))
    if not fuzz:
        return found, not_found_in_ref, not_found_in_inf

    fuzz_tokens_inf = []
    fuzz_tokens_ref = []

    for i in range(len(not_found_in_inf)):
        if include_role:
            token, role = not_found_in_inf[i]
        else:
            token, role = not_found_in_inf[i], None
        ft = FuzzyToken.from_actual_token(token, text, role)
        fuzz_tokens_inf.append(ft)

    for i in range(len(not_found_in_ref)):
        if include_role:
            token, role = not_found_in_ref[i]
        else:
            token, role = not_found_in_ref[i], None
        ft = FuzzyToken.from_actual_token(token, text, role)
        fuzz_tokens_ref.append(ft)

    found += FuzzyToken.list_intersection(fuzz_tokens_ref, fuzz_tokens_inf)
    not_found_in_ref = FuzzyToken.list_difference(fuzz_tokens_ref, fuzz_tokens_inf)
    not_found_in_inf = FuzzyToken.list_difference(fuzz_tokens_inf, fuzz_tokens_ref)
    # print("=" * 12)
    # print(text)
    # print("FOUND BY BOTH")
    # pprint.pp(found)
    # print("ONLY IN REF")
    # print(not_found_in_ref)
    # print("ONLY IN INF")
    # print(not_found_in_inf)
    return found, not_found_in_ref, not_found_in_inf,


def run_eval_ner(wdir: str, inference_folder, data_folder, cot, cre, include_role: bool = True, fuzz=False):
    logger.remove()
    wdir = os.path.join(_THIS_FOLDER, wdir)
    pathlib.Path(wdir).mkdir(parents=True, exist_ok=True)

    reaction_id_to_procedure_text = dict()
    test_data = json_load(f"{data_folder}/test.json")
    meta_data = json_load(f"{data_folder}/meta.json")
    for td, rid in zip(test_data, meta_data['test_data_identifiers']):
        reaction_id_to_procedure_text[rid] = td['instruction'].split("###")[1].strip()

    # load inferences
    pairs = get_pair_evaluators(inference_folder, data_folder, cot, cre)
    n_identical = 0
    n_instances = 0
    n_not_found_in_ref_sum = 0
    n_not_found_in_inf_sum = 0
    for pe in pairs:
        if not pe.valid_ord:
            continue
        compounds_inf = get_compounds(pe.reaction_message_inf, extracted_from="inputs")
        pcompounds_inf = get_compounds(pe.reaction_message_inf, extracted_from="outcomes")

        ner_inf = [get_ne(c, include_role) for c in compounds_inf + pcompounds_inf]
        if include_role:
            ner_inf = [ner for ner in ner_inf if ner[0] is not None]
        else:
            ner_inf = [ner for ner in ner_inf if ner is not None]
        ner_inf = sorted(set(ner_inf))
        compounds_ref = get_compounds(pe.reaction_message_ref, extracted_from="inputs")
        pcompounds_ref = get_compounds(pe.reaction_message_ref, extracted_from="outcomes")
        ner_ref = [get_ne(c, include_role) for c in compounds_ref + pcompounds_ref]
        ner_ref = sorted(set(ner_ref))

        found, not_found_in_ref, not_found_in_inf = count_found(ner_ref, ner_inf,
                                                                text=reaction_id_to_procedure_text[pe.reaction_id],
                                                                include_role=include_role, fuzz=fuzz)

        n_identical += len(found)
        n_instances += len(ner_ref)
        n_not_found_in_ref_sum += len(not_found_in_ref)
        n_not_found_in_inf_sum += len(not_found_in_inf)
    print(n_identical / n_instances, n_not_found_in_ref_sum / n_instances, n_not_found_in_inf_sum / n_instances)
    return n_identical, n_instances, n_not_found_in_ref_sum, n_not_found_in_inf_sum


if __name__ == '__main__':
    res = run_eval_ner(
        wdir="expt_202403020036_epoch14-cre-singular",
        inference_folder="expt_202403020036/epoch14-cre-singular",
        data_folder="../workplace_data/datasets/CRE_sinular",
        cot=False,
        include_role=False,
        cre=True,
        fuzz=True,
    )
    print(res)
