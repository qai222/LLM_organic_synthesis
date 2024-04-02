import json
import os.path

from google.protobuf import json_format
from loguru import logger
from ord_schema.proto.reaction_pb2 import Reaction

from ord.evaluation.evaluate_functions import evaluation_message_level, get_pair_evaluators, \
    evaluation_document_level, get_summary_table_message, evaluation_leaf_level, get_summary_table_leaf, FilePath

"""
This is the main script for evaluation
It takes 3 folders:
1. expt_202403020036/epoch14: output from the fine-tuned model for the uspto split 
2. expt_202403020036/epoch14-cre-singule: output from the fine-tuned model for cre unireaction dataset
3. ../workplace_cot/cot_response/USPTO-n100k-t2048_exp1-COT: output from gpt3.5 chain-of-thoughts 
and export evaluation results, one folder for each dataset
"""


_THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))


def main_evaluate(inference_folder: FilePath, data_folder: FilePath, wdir: FilePath, cot: bool, cre: bool):
    """
    main evaluation function

    :param inference_folder:
    :param data_folder:
    :param wdir:
    :param cot:
    :return:
    """
    message_level_csv = os.path.join(wdir, "message_level.csv")
    leaf_level_csv = os.path.join(wdir, "leaf_level.csv")
    message_level_log = os.path.join(wdir, "message_level.log")
    leaf_level_log = os.path.join(wdir, "leaf_level.log")
    doc_level_log = os.path.join(wdir, "document_level.log")
    message_level_summary_csv = os.path.join(wdir, "message_level_summary.csv")
    leaf_level_summary_strict_csv = os.path.join(wdir, "leaf_level_summary_strict.csv")
    leaf_level_summary_nonstrict_csv = os.path.join(wdir, "leaf_level_summary_nonstrict.csv")

    pairs = get_pair_evaluators(inference_folder, data_folder, cot, cre)

    doc_level_log_handle = logger.add(doc_level_log)
    repaired_pairs = evaluation_document_level(pairs)
    logger.remove(doc_level_log_handle)

    # ad-hoc fix for misplacing outcomes
    if cot:
        cot_invalid_json = 0
        cot_invalid_ord = 0
        for pe in repaired_pairs:
            inf_text = pe.inference_text
            inf_dict = json.loads(inf_text)
            if isinstance(inf_dict, dict):
                if "outcomes" in inf_dict['inputs']:
                    inf_dict['outcomes'] = inf_dict['inputs']['outcomes']
                    inf_dict['inputs'].pop('outcomes')
                inf_text = json.dumps(inf_dict)
            try:
                json.loads(inf_text)
                pe.valid_json = True
            except json.JSONDecodeError:
                cot_invalid_json += 1
            try:
                r = json_format.Parse(inf_text, Reaction())
                pe.valid_ord = True
            except json_format.ParseError:
                cot_invalid_ord += 1
            pe.inference_text = inf_text
        logger.critical(
            f"AFTER COT REPAIR\ntotal/invalid JSON/invalid ORD: {len(repaired_pairs)}/{cot_invalid_json}/{cot_invalid_ord}")

    message_level_log_handle = logger.add(message_level_log)
    df1 = evaluation_message_level(repaired_pairs, n_jobs=16)
    df1.to_csv(message_level_csv, index=False)
    sdf1 = get_summary_table_message(df1)
    sdf1.to_csv(message_level_summary_csv)
    logger.remove(message_level_log_handle)

    leaf_level_log_handle = logger.add(leaf_level_log)
    df2 = evaluation_leaf_level(repaired_pairs, n_jobs=16)
    df2.to_csv(leaf_level_csv, index=False)
    sdf2 = get_summary_table_leaf(df2, strict=True)
    sdf2.to_csv(leaf_level_summary_strict_csv)
    sdf2 = get_summary_table_leaf(df2, strict=False)
    sdf2.to_csv(leaf_level_summary_nonstrict_csv)
    logger.remove(leaf_level_log_handle)


if __name__ == '__main__':
    main_evaluate(
        inference_folder="expt_202403020036/epoch14",
        data_folder="../workplace_data/datasets/USPTO-n100k-t2048_exp1",
        wdir="expt_202403020036_epoch14",
        cot=False,
        cre=False,
    )

    main_evaluate(
        inference_folder="expt_202403020036/epoch14-cre-singular",
        data_folder="../workplace_data/datasets/CRE_sinular",
        wdir="expt_202403020036_epoch14-cre-singular",
        cot=False,
        cre=True,
    )

    main_evaluate(
        inference_folder="../workplace_cot/cot_response/USPTO-n100k-t2048_exp1-COT",
        data_folder="../workplace_data/datasets/USPTO-n100k-t2048_exp1-COT.json",
        wdir="gpt3.5_cot",
        cot=True,
        cre=False,
    )
