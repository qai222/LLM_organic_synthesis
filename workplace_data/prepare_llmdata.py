import os.path

from loguru import logger

from ord.data.data_llm import LlmDataset, DEFAULT_PROMPT_TEMPLATE
from ord.utils import json_dump, json_load


def prepare_USPTO_master():
    """ this function produces a master `LlmDataset` for all USPTO reactions """
    master_file = "USPTO-master.json.gz"
    if os.path.isfile(master_file):
        logger.warning("Skip creating master file as the master file already exists: `USPTO-master.json.gz`")
        master_dataset = LlmDataset(**json_load(master_file))
        logger.info(f"# of reactions in the master dataset: {len(master_dataset.data)}")  # 1339260
        return master_dataset
    master_dataset = LlmDataset(
        source_file="uspto/export_from_pb_dedup.json.gz",
        prompt_template=DEFAULT_PROMPT_TEMPLATE,
        ord_root_fields=['inputs', 'conditions', 'workups', 'outcomes'],
        tokenizer_path="7B_tokenizer/tokenizer.model",
        data=[]
    )
    master_dataset.load()
    master_dataset_dump = master_dataset.model_dump()
    json_dump(master_file, master_dataset_dump, gz=True)
    return master_dataset


def make_USPTO_split(master_dataset: LlmDataset, total_size: int, n_token_limit: int, name: str = None):
    """
    1. make a cdf for the sentence lengths in master dataset
    2. apply a token limit (max)
    3. random sample from the remaining

    :param master_dataset:
    :param total_size:
    :param n_token_limit:
    :param name:
    :return:
    """
    fig = master_dataset.plot_n_token_cdf(n_token_limit=n_token_limit)
    fig.savefig("cdf.png", dpi=600)
    if name is None:
        name = f"USPTO-n{total_size // 1000}k-t{n_token_limit}"
    master_dataset.split(
        total_size=total_size,
        n_token_limit=n_token_limit,
        name=name,
    )


if __name__ == '__main__':
    USPTO_MASTER = prepare_USPTO_master()
    make_USPTO_split(USPTO_MASTER, 100 * 1000, 2048)
