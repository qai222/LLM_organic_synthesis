import os.path
import pathlib
from collections import defaultdict

import pandas as pd

from ord_diff import json_load, json_dump
from ord_diff.evaluation import BatchEvaluator, logger

_THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))


def get_summary_table_leaf(leaf_level_df: pd.DataFrame, strict: bool):
    df = leaf_level_df
    table_data = defaultdict(dict)

    unique_leaf_types =

    n_m1_fields = 0
    for record in df.to_dict(orient="records"):
        lt = record['leaf_type']
        ct = record['change_type']
        if ct is None or pd.isna(ct):
            ct = "intact"
        if not strict and not record['considered_in_nonstrict']:
            continue
        try:
            table_data[lt][ct] += 1
        except KeyError:
            table_data[lt][ct] = 1
        if record["from"] == "m1":
            n_m1_fields += 1
    df = pd.DataFrame(table_data)
    df = df.fillna(0).transpose()
    print(f"total fields: {n_m1_fields}")
    print(f"sum of intact, removal, and alteration: {df['intact'].sum() + df['REMOVAL'].sum() + df['ALTERATION'].sum()}")
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
        pct = "{:.2%}".format(v / ref_dict[mt])
        table_data[mt][attr_name] = f"{v} ({pct})"
    summary_df = pd.DataFrame(data=table_data)
    return summary_df.transpose()


def run_eval(expt_name: str, level: str):
    logger.remove()
    wdir = os.path.join(_THIS_FOLDER, expt_name)
    pathlib.Path(wdir).mkdir(parents=True, exist_ok=True)

    # load inferences
    batch_evaluator_json = os.path.join(wdir, "batch_evaluator.json.gz")
    if os.path.isfile(batch_evaluator_json):
        batch_evaluator = BatchEvaluator(**json_load(batch_evaluator_json))
    else:
        logger.add(os.path.join(wdir, "batch_evaluator.log"))
        batch_evaluator = BatchEvaluator.load_inferences(
            inference_folder=f"{_THIS_FOLDER}/../models_llama/adapter/infer-expt_{expt_name}",
            data_folder=f"{_THIS_FOLDER}/../models_llama/adapter/USPTO-t900",
        )
        json_dump(batch_evaluator_json, batch_evaluator.model_dump(), gz=True)
        logger.remove()

    if level == "message":
        # get message level eval csv
        message_level_csv = os.path.join(wdir, "message_level.csv")
        if os.path.isfile(message_level_csv):
            df = pd.read_csv(message_level_csv)
        else:
            logger.add(os.path.join(wdir, "message_level.log"))
            df = batch_evaluator.eval_message_level()
            df.to_csv(message_level_csv, index=False)
            logger.remove()

        # to summary table
        summary_table = os.path.join(wdir, "message_level_summary.csv")
        summary_table_df = get_summary_table_message(df)
        summary_table_df.to_csv(summary_table)

    elif level == "leaf":
        # get leaf level eval csv
        leaf_level_csv = os.path.join(wdir, "leaf_level.csv")
        if os.path.isfile(leaf_level_csv):
            df = pd.read_csv(leaf_level_csv)
        else:
            logger.add(os.path.join(wdir, "leaf_level.log"))
            df = batch_evaluator.eval_leaf_level()
            df.to_csv(leaf_level_csv, index=False)
            logger.remove()

        # to summary table
        summary_table_strict = os.path.join(wdir, "leaf_level_summary_strict.csv")
        get_summary_table_leaf(df, strict=True).to_csv(summary_table_strict)
        summary_table_nonstrict = os.path.join(wdir, "leaf_level_summary_nonstrict.csv")
        get_summary_table_leaf(df, strict=False).to_csv(summary_table_nonstrict)


if __name__ == '__main__':
    llama1_t900_expt_name, llama2_t900_expt_name = "202311060037", "202311062152"
    run_eval(llama1_t900_expt_name, level="message")
    run_eval(llama1_t900_expt_name, level="leaf")
    run_eval(llama2_t900_expt_name, level="message")
    run_eval(llama2_t900_expt_name, level="leaf")
