import os.path

from loguru import logger

from ord.evaluation.evaluate_functions import evaluation_message_level, get_pair_evaluators, \
    evaluation_document_level, get_summary_table_message, evaluation_leaf_level, get_summary_table_leaf, FilePath

_THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))


def main_evaluate(inference_folder: FilePath, data_folder: FilePath, wdir: FilePath, cot: bool):
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

    pairs = get_pair_evaluators(inference_folder, data_folder, cot)

    doc_level_log_handle = logger.add(doc_level_log)
    repaired_pairs = evaluation_document_level(pairs)
    logger.remove(doc_level_log_handle)

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
        inference_folder="expt_202402171922/epoch14",
        data_folder="../workplace_data/datasets/USPTO-t100k-n2048",
        wdir="expt_202402171922_epoch14",
        cot=False
    )
