from __future__ import annotations

import pandas as pd
from loguru import logger
from pandas._typing import FilePath
from tqdm import tqdm

from .pair_evaluator import PairEvaluator


def get_pair_evaluators(inference_folder: FilePath, dataset_folder: FilePath, cot=False):
    if cot:
        return PairEvaluator.from_inference_folder(inference_folder, dataset_folder)
    else:
        return PairEvaluator.from_inference_folder_cot(inference_folder, dataset_folder)


def evaluation_leaf_level(pair_evaluators: list[PairEvaluator]) -> pd.DataFrame:
    dfs = []
    for pe in tqdm(pair_evaluators, desc="batch evaluation -- leaf level"):
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


def evaluation_message_level(pair_evaluators: list[PairEvaluator]) -> pd.DataFrame:
    records = []
    n_invalid_json = 0
    n_invalid_ord = 0
    for pe in tqdm(pair_evaluators, desc="batch evaluation -- message level"):
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
    logger.critical(f"total/invalid JSON/invalid ORD: {len(pair_evaluators)}/{n_invalid_json}/{n_invalid_ord}")
    return pd.DataFrame.from_records(records)
