import glob
import json
import random
from typing import Any

import tqdm

"""
alpaca training data looks like the following
{
    "instruction": <a string sentence about what should be done given the `input`>,
    "input": <string>,
    "output": <string>,
}
"""
ORD_DATASET_FILES = sorted(glob.glob("data_from_pb_no_warning_20230416/data_from_pb_no_warning/*.json"))
ORD_PROCEDURE_FIELD = "notes__procedureDetails"
ALPACA_INSTRUCTION = "Extract a JSON of reaction inputs using ORD data schema"


def reaction_to_alpaca(r: dict[str, Any]):
    d = {
        "reaction_id": r['reaction_id'],
        "instruction": ALPACA_INSTRUCTION,
        "input": r[ORD_PROCEDURE_FIELD],
        "output": str(r['inputs']),
    }
    return d


def create_datasets(input_length=500, output_length=500, dataset_size=1000, train_test_ratio=0.2, seed=42):
    reactions = []
    for jp in tqdm.tqdm(ORD_DATASET_FILES):
        with open(jp, "r") as f:
            batch_reactions = json.load(f)
            batch_reactions = [
                r for r in batch_reactions
                if len(str(r['inputs'])) < input_length and len(str(r[ORD_PROCEDURE_FIELD])) < output_length
            ]
            reactions += batch_reactions
    random.seed(seed)
    reactions = random.sample(reactions, int(dataset_size * (1 + train_test_ratio)))
    random.shuffle(reactions)
    alpaca_data = [reaction_to_alpaca(r) for r in reactions]
    train, test = alpaca_data[:dataset_size], alpaca_data[dataset_size:]
    return train, test


def save_datasets(input_length=500, output_length=500, dataset_size=1000, train_test_ratio=0.2, seed=42,
                  name="alpaca_data"):
    train, test = create_datasets(input_length, output_length, dataset_size, train_test_ratio, seed)

    with open(f"{name}_train.json", "w") as f:
        json.dump(train, f)

    with open(f"{name}_test.json", "w") as f:
        json.dump(test, f)


if __name__ == '__main__':
    save_datasets(
        input_length=1000,
        output_length=1000,
        dataset_size=1000,
        train_test_ratio=0.2,
        seed=42,
        name="alpaca_1000"
    )
