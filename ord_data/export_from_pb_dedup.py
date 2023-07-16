import glob
import json

from tqdm import tqdm

"""
There are duplicate entries in the outputs from `export_from_pb.py`. These can come from upstream USPTO data.
"""


def dup_id():
    all_reactions = []
    for jp in tqdm(sorted(glob.glob("../data_from_pb_no_warning_20230416/data_from_pb_no_warning/ord_dataset-*.json"))):
        with open(jp, "r") as f:
            batch_reactions = json.load(f)
        all_reactions += batch_reactions

    all_reactions_openai = []
    for r in all_reactions:
        prompt = r['notes__procedureDetails'] + "\n\n###\n\n."
        all_reactions_openai.append(
            {
                "prompt": prompt,
                "completion": " blabla",
            }
        )

    with open("data_from_pb_no_warning_20230416.json", "w") as f:
        json.dump(all_reactions, f)
    with open("data_from_pb_no_warning_20230416_dedup_tmp.json", "w") as f:
        json.dump(all_reactions_openai, f)


def dup_remove():
    """ `openai tools fine_tunes.prepare_data -f data_from_pb_no_warning_20230416_dedup_tmp.json`  """
    openai_prompts = []
    with open("data_from_pb_no_warning_20230416_dedup_tmp_prepared.jsonl", "r") as f:
        lines = f.readlines()
    for line in lines:
        p = json.loads(line)['prompt']
        openai_prompts.append(p)

    with open("data_from_pb_no_warning_20230416.json", "r") as f:
        all_reactions = json.load(f)

    unique_reactions = []
    unique_details = []
    for r in all_reactions:
        detail = r['notes__procedureDetails']
        prompt = detail + "\n\n###\n\n."
        if prompt in openai_prompts and detail not in unique_details:
            unique_reactions.append(r)
            unique_details.append(detail)
    print(len(unique_reactions))
    with open("data_from_pb_no_warning_20230416_dedup.json", "w") as f:
        json.dump(unique_reactions, f)


if __name__ == '__main__':
    dup_id()
    dup_remove()
