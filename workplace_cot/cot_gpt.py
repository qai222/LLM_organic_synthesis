import pprint
import random
from pathlib import Path

import tiktoken
import tqdm
from openai import OpenAI
from pandas._typing import FilePath

from ord.utils import json_load, json_dump

_API_KEY = "YOUR_API_KEY"
Client = OpenAI(api_key=_API_KEY)
MODEL_NAME = "gpt-3.5-turbo-0125"
MAX_TOKENS = 3000


def print_models():
    ml = Client.models.list()
    pprint.pp([*ml])


def sample_test_set(test_json: FilePath, k=100):
    test_set = json_load(test_json)
    random.seed(42)
    test_set = random.sample(test_set, k)
    return sorted(test_set, key=lambda x: x['reaction_id'])


def get_cot_prompt(procedure_text: str, cot_prefix_file: str = "cot_prefix.txt"):
    with open(cot_prefix_file, "r") as f:
        cot_prefix = f.read()

    prompt = f"""{cot_prefix}
Here is a new reacton_text.
new_reaction_text = ```{procedure_text}```
Please follow the above instructions and the workflow for dealing the two given examples. Extract and return the ORD JSON record."""
    return prompt


def calculate_tokens(prompt: str):
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(prompt))


def get_response(procedure_text: str):
    prompt = get_cot_prompt(procedure_text)
    response = Client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "Act as a professional researcher in organic chemistry."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=MAX_TOKENS,
        # max_tokens=TOKEN_LIMIT - calculate_tokens(prompt),
        top_p=1
    )
    return response


def cot_experiment(test_json: FilePath, k: int, dump_folder: FilePath):
    test_set_samples = sample_test_set(test_json, k)
    Path(dump_folder).mkdir(parents=True, exist_ok=True)
    for data in tqdm.tqdm(test_set_samples):
        reaction_text = data['procedure_text']
        reaction_id = data['reaction_id']
        dump_file = f"{dump_folder}/{reaction_id}.json"
        # if os.path.isfile(dump_file):
        #     if json_load(dump_file)['choices'][0]['finish_reason'] == 'stop':
        #         continue
        #     else:
        #         logger.warning(f"found a unfinished call: {reaction_id}, retrying")
        gpt_response = get_response(reaction_text)
        json_dump(dump_file, gpt_response.model_dump(), indent=2)


if __name__ == '__main__':
    cot_experiment(
        test_json="/home/qai/workplace/LLM_organic_synthesis/workplace_data/datasets/USPTO-n100k-t2048_exp1-COT.json",
        k=500,
        dump_folder="/home/qai/workplace/LLM_organic_synthesis/workplace_cot/cot_response/USPTO-n100k-t2048_exp1-COT"
    )
