import tiktoken
import random

import tqdm
from openai import OpenAI
from ord.utils import json_load, json_dump
from pandas._typing import FilePath
import pprint

_API_KEY = "YOUR_API_KEY"
Client = OpenAI(api_key=_API_KEY)
MODEL_NAME = "gpt-3.5-turbo-0125"
MAX_TOKENS = 1000

def print_models():
    ml = Client.models.list()
    pprint.pp([*ml])

def sample_test_set(dataset_path: FilePath, k=100):
    test_json = f"{dataset_path}/test.json"
    meta_json = f"{dataset_path}/meta.json"
    test_set = json_load(test_json)
    test_ids = json_load(meta_json)['test_data_identifiers']
    assert len(test_set) == len(test_ids)
    test_set = [{"reaction_id": i, **d} for d, i in zip(test_set, test_ids)]
    random.seed(42)
    test_set = random.sample(test_set, k)
    return sorted(test_set, key=lambda x:x['reaction_id'])

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


def get_response(procedure_text:str):
    response = Client.chat.completions.create(
      model = "gpt-3.5-turbo-0125",
      messages=[
        {
          "role": "system",
          "content": "Act as a professional researcher in organic chemistry."
        },
        {
          "role": "user",
          "content": get_cot_prompt(procedure_text)
        }
      ],
      temperature=0,
      max_tokens=MAX_TOKENS,
      top_p=1
    )
    return response

if __name__ == '__main__':
    test_set_samples = sample_test_set(
        dataset_path="/home/qai/workplace/LLM_organic_synthesis/workplace_data/datasets/USPTO-t100k-n2048",
        k=100
    )
    for data in tqdm.tqdm(test_set_samples[:5]):
        reaction_text = data['instruction']
        reaction_id = data['reaction_id']
        p = get_cot_prompt(reaction_text)
        gpt_response = get_response(reaction_text)
        json_dump(f"cot_response/{reaction_id}.json", gpt_response.model_dump(), indent=2)
