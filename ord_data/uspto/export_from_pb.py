import glob
import json
import os.path

from loguru import logger
from ord_schema.message_helpers import load_message
from ord_schema.proto.dataset_pb2 import Dataset
from tqdm import tqdm

from data_schema import LDataExportError, reaction_to_llm_data

"""
filed name in the output of `protobuf.json_format` is by default camel case
use `json_format.MessageToJson(r, preserving_proto_field_name=True)` to output unmodified keys
"""


def convert_datasets(
        output_data_dir="data_from_pb_no_warning",
        local_data_folder="/home/qai/workplace/ord-data/data",
        uspto_only=True,
        keep_only_no_warning=True,
):
    dataset_files = sorted(glob.glob(f"{local_data_folder}/*/*.pb.gz"))
    logger.info(f"# of dataset files: {len(dataset_files)}")
    for f in tqdm(dataset_files):
        logger.info(f"convert dataset file: {f}")
        dataset = load_message(f, Dataset)
        if uspto_only and "uspto-grants-" not in dataset.name.lower():
            logger.critical(f"skipping non-USPTO dataset: {dataset.name}")
        output_file = f"{output_data_dir}/{dataset.dataset_id}.json"
        if os.path.isfile(output_file) and os.path.getsize(output_file) > 0:
            logger.info("output already exists, skipping conversion...")
            continue
        lds = []
        for r in dataset.reactions:
            try:
                ld = reaction_to_llm_data(r)
            except LDataExportError as e:
                logger.critical(f"excluding reaction: {r.reaction_id}\ndue to:{e}")
                continue
            if keep_only_no_warning and len(ld['warning_messages']) > 0:
                logger.critical(f"excluding reaction: {r.reaction_id}\ndue to warning msgs: {ld['warning_messages']}")
                continue
            lds.append(ld)
        with open(output_file, 'w') as output_fp:
            json.dump(lds, output_fp, indent=2)


if __name__ == '__main__':
    logger.remove(0)
    logger.add(__file__.replace(".py", ".log"))
    convert_datasets(keep_only_no_warning=True)
    # 190764 reactions 515 datasets
