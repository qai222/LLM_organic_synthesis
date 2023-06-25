import json
import os.path

import openai
from tqdm import tqdm

openai.api_key = '<API_KEY>'


def get_completion(prompt):
    prompt += "\n\n###\n\n"
    inference = openai.Completion.create(
        # model="curie:ft-llm-hackathon-synthesis-parsing-team-2023-03-29-22-57-05",
        model="davinci:ft-llm-hackathon-synthesis-parsing-team-2023-03-30-00-54-58",
        prompt=prompt,
        max_tokens=1000,
        stop=["###"]
    )
    completion_text = inference["choices"][0]["text"]
    return completion_text


this_dir = os.path.dirname(__file__)
train_data_path = os.path.join(this_dir, "../dash_app/assets/data1000_v2.json")
test_data_path = os.path.join(this_dir, "../dash_app/assets/data50_v2_test.json")

with open(train_data_path, "r") as f:
    train_data = json.load(f)

with open(test_data_path, "r") as f:
    test_data = json.load(f)

train_data_ids = set([r['reaction_id'] for r in train_data])

test_results = []
for d in tqdm(test_data):
    input_text = d['input_text']
    completion_text = get_completion(input_text)
    result = {
        "data": d,
        "completion": completion_text,
    }
    test_results.append(result)

with open('test_results.json', "w") as f:
    json.dump(test_results, f)
