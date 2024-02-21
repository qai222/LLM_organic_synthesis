import glob
import json
import os.path

from loguru import logger
from ord_schema.message_helpers import load_message
from ord_schema.proto import dataset_pb2
from tqdm import tqdm

from ord.data_reaction import ReactionDataError, ReactionData
from ord.utils import json_load, json_dump

"""
use this to export data from dataset pb.gz to json files downloaded from ord-data repository
"""

USPTO_ONLY = True
""" only export from USPTO datasets """

LOCAL_DATA_FOLDER = "/home/qai/workplace/ord-data/data"
""" this should have the structure of <FOLDER>/*/*.pb.gz """

OUTPUT_DIR = os.path.dirname(__file__) + "/export_from_pb/"
""" where to store outputs """

if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


def export_from_dataset_pb(dataset_pb_path):
    logger.info(f"convert dataset file: {dataset_pb_path}")
    dataset = load_message(dataset_pb_path, dataset_pb2.Dataset)

    if USPTO_ONLY and "uspto-grants-" not in dataset.name.lower():
        logger.critical(f"skipping non-USPTO dataset: {dataset.name}")
        return

    output_file = f"{OUTPUT_DIR}/{dataset.dataset_id}.json"
    if os.path.isfile(output_file) and os.path.getsize(output_file) > 0:
        logger.info("output already exists, skipping")
        return

    reaction_data_list = []
    for r in dataset.reactions:
        try:
            ld = ReactionData.from_reaction_message(r)
        except ReactionDataError as e:
            logger.critical(f"excluding reaction: {r.reaction_id}\ndue to:{e}")
            continue
        reaction_data_list.append(ld.model_dump())
    json_dump(output_file, reaction_data_list, indent=2)
    return reaction_data_list


def export_datasets():
    """
    export datasets from *.pb files to json

    :return:
    """
    dataset_files = sorted(glob.glob(f"{LOCAL_DATA_FOLDER}/*/*.pb.gz"))
    logger.info(f"# of dataset files: {len(dataset_files)}")
    for f in tqdm(dataset_files):
        export_from_dataset_pb(f)


def deduplicate_pre(export_from_pb_folder: str = "export_from_pb"):
    """
    preprocess for deduplication

    :param export_from_pb_folder: a folder of ord-dataset*.json files
    :return:
    """
    reactions = []
    n_datasets = 0
    for jp in tqdm(sorted(glob.glob(f"{export_from_pb_folder}/ord_dataset-*.json"))):
        batch_reactions = json_load(jp)
        reactions += batch_reactions
        n_datasets += 1

    logger.info(f"prepare to deduplicate from {n_datasets} datasets")
    reactions_openai = []
    for r in reactions:
        prompt = r['procedure_text'] + "\n\n###\n\n."
        reactions_openai.append(
            {
                "prompt": prompt,
                "completion": " blabla",
            }
        )
    json_dump("export_from_pb.json.gz", reactions, gz=True)
    json_dump("export_from_pb_dedup_tmp.json", reactions_openai)


def deduplicate(output_jsonl: str = "export_from_pb_dedup_tmp_prepared.json"):
    """
    use openai tools to deduplicate
    `openai tools fine_tunes.prepare_data -f export_from_pb_dedup_tmp.json`

    making the following selections
    - [Recommended] Remove 431772 duplicate rows [Y/n]: y
    - [Recommended] Remove 65 long examples [Y/n]: n
    - [Recommended] Would you like to split into training and validation set? [Y/n]: n

    :param output_jsonl:
    :return:
    """
    if not os.path.isfile(output_jsonl):
        logger.error(f"Cannot find: {output_jsonl}, did you run openai tools?")
        return

    unique_openai_prompts = []
    with open(output_jsonl, "r") as f:
        lines = f.readlines()
    for line in lines:
        p = json.loads(line)['prompt']
        unique_openai_prompts.append(p)

    reactions = json_load("export_from_pb.json.gz")

    logger.info(f"# of reactions before deduplication: {len(reactions)}")  # 1771032

    openai_prompt_to_reaction = dict()
    for r in tqdm(reactions, desc="build openai prompt lookup table"):
        procedure_text = r['procedure_text']
        openai_prompt = procedure_text + "\n\n###\n\n."
        openai_prompt_to_reaction[openai_prompt] = r

    unique_reactions = []
    for unique_openai_prompt in tqdm(unique_openai_prompts, desc="add unique reactions"):
        unique_reaction = openai_prompt_to_reaction[unique_openai_prompt]
        unique_reactions.append(unique_reaction)
    logger.info(f"# of reactions after deduplication: {len(unique_reactions)}")   # 1339260
    json_dump("export_from_pb_dedup.json.gz", unique_reactions, gz=True)


if __name__ == '__main__':
    logger.remove(0)
    logger.add(__file__.replace(".py", ".log"))
    export_datasets()
    deduplicate_pre("export_from_pb")
    deduplicate(output_jsonl = "export_from_pb_dedup_tmp_prepared.jsonl")
