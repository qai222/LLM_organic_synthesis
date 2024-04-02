import os.path

import pandas as pd
from google.protobuf.json_format import Parse
from ord_schema.message_helpers import find_submessages
from ord_schema.proto.reaction_pb2 import Reaction, Compound, ProductCompound
from tqdm import tqdm

from ord.evaluation.evaluate_functions import FilePath
from ord.utils import DeepDiffKey, DeepDiff
from ord.utils import json_load

_THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))


def main_evaluate_cde(cde_results: FilePath, dataset_folder: FilePath):
    """
    main evaluation function

    :param cde_results:
    :param data_folder:
    :param wdir:
    :param cot:
    :return:
    """
    test_data = json_load(f"{dataset_folder}/test.json")
    test_ids = json_load(f"{dataset_folder}/meta.json")['test_data_identifiers']
    cde_chemical_lists = json_load(cde_results)
    records = []
    for tid, data in tqdm(zip(test_ids, test_data)):
        reaction = Parse(data['output'], Reaction())
        compounds = find_submessages(reaction, submessage_type=Compound)
        compounds += find_submessages(reaction, submessage_type=ProductCompound)
        chemical_list_ref = [c.identifiers[0].value for c in compounds if len(c.identifiers)]
        chemical_list_ner = cde_chemical_lists[tid]
        diff = DeepDiff(chemical_list_ref, chemical_list_ner, ignore_order=True)
        try:
            addition = diff[DeepDiffKey.iterable_item_added]
        except KeyError:
            addition = []
        try:
            removal = diff[DeepDiffKey.iterable_item_removed]
        except KeyError:
            removal = []
        try:
            alteration = diff[DeepDiffKey.values_changed]
        except KeyError:
            alteration = []
        record = {
            # "reaction_id": tid,
            "n_ref": len(chemical_list_ref),
            "n_inf": len(chemical_list_ner),
            "addition": len(addition),
            "removal": len(removal),
            "alteration": len(alteration),
            "intact": len(chemical_list_ref) - len(removal) - len(alteration),
        }
        records.append(record)
    return pd.DataFrame.from_records(records)


if __name__ == '__main__':
    df = main_evaluate_cde("cde.json", "../workplace_data/datasets/USPTO-n100k-t2048_exp1/")
    df.to_csv("cde.csv", index=False)
    print(df.sum(axis=0))
    df.sum(axis=0).to_csv("cde_sum.csv", index=False)

    df = main_evaluate_cde("llm.json", "../workplace_data/datasets/USPTO-n100k-t2048_exp1/")
    df.to_csv("llm_cde.csv", index=False)
    print(df.sum(axis=0))
    df.sum(axis=0).to_csv("llm_cde_sum.csv", index=False)
