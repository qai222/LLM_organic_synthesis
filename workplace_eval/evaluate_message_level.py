import os.path
import pathlib

import pandas as pd

from ord_diff import json_load, json_dump
from ord_diff.evaluation import BatchEvaluator, logger

_THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))


def get_summary_table(message_level_df):
    from collections import defaultdict
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
    return summary_df


def run_eval(expt_name: str):
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
    summary_table_df = get_summary_table(df)
    summary_table_df.to_csv(summary_table)


if __name__ == '__main__':
    llama1_t900_expt_name, llama2_t900_expt_name = "202311060037", "202311062152"
    run_eval(llama1_t900_expt_name)
