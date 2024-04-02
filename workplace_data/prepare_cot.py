import random

from pandas._typing import FilePath

from ord.data.data_llm import prompt_to_procedure
from ord.utils import json_load, json_dump

BASE_DATASET = "datasets/USPTO-n100k-t2048_exp1"


def create_dataset_for_cot(base_dataset: FilePath, size: int = 500):
    test_ids = json_load(f"{base_dataset}/meta.json")['test_data_identifiers']
    test_data = json_load(f"{base_dataset}/test.json")
    test_dict = dict(zip(test_ids, test_data))
    random.seed(42)
    cot_data = []
    for test_id in random.sample(test_ids, size):
        ref_string = test_dict[test_id]['output']
        procedure_text = prompt_to_procedure(test_dict[test_id]['instruction'])
        record = {
            "reaction_id": test_id,
            "procedure_text": procedure_text,
            "reference_string": ref_string,
        }
        cot_data.append(record)
    json_dump(f"{base_dataset}-COT.json", cot_data, indent=2)
    return cot_data


if __name__ == '__main__':
    create_dataset_for_cot(BASE_DATASET, 1000)