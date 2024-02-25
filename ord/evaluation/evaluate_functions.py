from __future__ import annotations

from collections import defaultdict

import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from pandas._typing import FilePath
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from .pair_evaluator import PairEvaluator, DeltaType


def get_pair_evaluators(inference_folder: FilePath, dataset_folder: FilePath, cot=False):
    if not cot:
        return PairEvaluator.from_inference_folder(inference_folder, dataset_folder)
    else:
        return PairEvaluator.from_inference_folder_cot(inference_folder, dataset_folder)


def _evaluation_leaf_level(pe: PairEvaluator) -> pd.DataFrame | None:
    if pe.valid_ord:
        try:
            df = pe.eval_leaf_level_all()
        except Exception as e:
            logger.critical(f"evaluation error for: {pe.reaction_id}\n{e.__str__()}")
            return
        if df is None:
            logger.critical(f"timeout error for: {pe.reaction_id}")
        else:
            return df


def evaluation_leaf_level(pair_evaluators: list[PairEvaluator], n_jobs=10) -> pd.DataFrame:
    with tqdm_joblib(tqdm(desc="batch evaluation -- leaf level", total=len(pair_evaluators))) as progress_bar:
        dfs = Parallel(n_jobs=n_jobs)(delayed(_evaluation_leaf_level)(pe) for pe in pair_evaluators)
    dfs = [df for df in dfs if df is not None]
    return pd.concat(dfs, axis=0)


def evaluation_document_level(pair_evaluators: list[PairEvaluator]):
    repaired = []
    for pe in tqdm(pair_evaluators, desc="batch evaluation -- document level"):
        if pe.valid_json:
            rpe = pe
        else:
            rpe = pe.repair()
        repaired.append(rpe)

    n_invalid_json = len([pe for pe in pair_evaluators if not pe.valid_json])
    n_invalid_ord = len([pe for pe in pair_evaluators if not pe.valid_ord])
    n_invalid_json_repair = len([pe for pe in repaired if not pe.valid_json])
    n_invalid_ord_repair = len([pe for pe in repaired if not pe.valid_ord])
    logger.critical(
        f"BEFORE REPAIR\ntotal/invalid JSON/invalid ORD: {len(pair_evaluators)}/{n_invalid_json}/{n_invalid_ord}")
    logger.critical(
        f"AFTER  REPAIR\ntotal/invalid JSON/invalid ORD: {len(pair_evaluators)}/{n_invalid_json_repair}/{n_invalid_ord_repair}")
    return repaired


def _evaluation_message_level(pe: PairEvaluator) -> dict | None:
    if pe.valid_ord:
        try:
            record = pe.eval_message_level_all()
        except Exception as e:
            logger.critical(f"evaluation error for: {pe.reaction_id}\n{e.__str__()}")
            return None
        if record is None:
            logger.critical(f"timeout error for: {pe.reaction_id}")
            return None
        else:
            return record


def evaluation_message_level(pair_evaluators: list[PairEvaluator], n_jobs=10) -> pd.DataFrame:
    with tqdm_joblib(tqdm(desc="batch evaluation -- message level", total=len(pair_evaluators))) as progress_bar:
        records = Parallel(n_jobs=n_jobs)(delayed(_evaluation_message_level)(pe) for pe in pair_evaluators)
    records = [r for r in records if r is not None]
    return pd.DataFrame.from_records(records)


def get_summary_table_leaf(leaf_level_df: pd.DataFrame, strict: bool):
    df = leaf_level_df
    table_data = defaultdict(dict)

    unique_leaf_types = set(df['leaf_type'].tolist())
    unique_change_types = set(df['change_type'].tolist())
    for lt in unique_leaf_types:
        for ct in unique_change_types:
            if ct is None or pd.isna(ct):
                ct = "intact"
            table_data[lt][ct] = 0

    n_m1_fields = 0
    for record in df.to_dict(orient="records"):
        lt = record['leaf_type']
        ct = record['change_type']
        if ct is None or pd.isna(ct):
            ct = "intact"
        if not strict and not record['considered_in_nonstrict']:
            continue
        table_data[lt][ct] += 1
        if record["from"] == "m1":
            n_m1_fields += 1

    df = pd.DataFrame(table_data)
    df = df.fillna(0).transpose()
    print(f"total fields: {n_m1_fields}")
    print(
        f"sum of intact, removal, and alteration: {df['intact'].sum() + df['REMOVAL'].sum() + df['ALTERATION'].sum()}")
    df["total m1"] = df[[DeltaType.REMOVAL, DeltaType.ALTERATION, 'intact']].sum(axis=1)
    pdf = df.apply(lambda x: x / x[-1], axis=1)
    h, w = df.shape
    table = defaultdict(dict)
    for i in range(h):
        for j in range(w):
            table[i][j] = str(df.values[i][j]) + " " + "({:.2%})".format(pdf.values[i][j])
    for i in range(h):
        df.iloc[i, :] = list(table[i].values())
    return df


def get_summary_table_message(message_level_df: pd.DataFrame):
    summary = message_level_df.sum(axis=0)
    summary: pd.Series
    table_data = defaultdict(dict)

    ref_dict = dict()
    for k, v in summary.to_dict().items():
        if "__" not in k:
            continue
        mt, attr_name = k.split("__")
        if attr_name == "n_ref":
            ref_dict[mt] = v

    for k, v in summary.to_dict().items():
        if "__" not in k:
            continue
        mt, attr_name = k.split("__")
        try:
            pct = "{:.2%}".format(v / ref_dict[mt])
        except ZeroDivisionError:
            pct = "N/A"
        table_data[mt][attr_name] = f"{v} ({pct})"
    summary_df = pd.DataFrame(data=table_data)
    return summary_df.transpose()
