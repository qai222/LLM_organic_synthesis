import glob
import json
import random
from typing import Any

import tqdm

"""
alpaca training data follows the following schema

{
    "instruction": <a string sentence about what should be done given the `input`>,
    "input": <string>,
    "output": <string>,
}
"""

ord_dataset_files = sorted(glob.glob("data_from_pb_no_warning_20230416/data_from_pb_no_warning/*.json"))
ord_procedure_field = "notes__procedureDetails"


def reaction_to_alpaca(r: dict[str, Any]):
    d = {
        "instruction": "extract a valid JSON describing reaction inputs from this organic synthesis procedure",
        "input": r[ord_procedure_field],
        "output": r['inputs'],
    }
    return d


reactions = []
for jp in tqdm.tqdm(ord_dataset_files):
    with open(jp, "r") as f:
        reactions += json.load(f)

random.seed(42)
reactions = random.sample(reactions, 300)
alpaca_data = [reaction_to_alpaca(r) for r in reactions]

with open(f"alpaca_300.json", "w") as f:
    json.dump(alpaca_data, f)
