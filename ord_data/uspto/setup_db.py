"""
1. setup rdkit psql for ord
```
conda install -c rdkit rdkit-postgresql
mkdir -p "${HOME}/rdkit-postgresql-for-ord"
export PGDATA="${HOME}/rdkit-postgresql-for-ord"
initdb -U $USERNAME
pg_ctl -D /home/$USERNAME/rdkit-postgresql-for-ord -l logfile start
createdb ord
```
2. load data to db using this script
"""
import glob

import pandas as pd
from loguru import logger
from ord_schema.message_helpers import fetch_dataset, load_message
from ord_schema.orm.database import add_dataset, add_rdkit
from ord_schema.orm.database import prepare_database
from ord_schema.orm.mappers import _MESSAGE_TO_MAPPER
from ord_schema.proto.dataset_pb2 import Dataset
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from tqdm import tqdm

CONNECTING_STRING = "postgresql://127.0.0.1:5432/ord"
LOCAL_DATA_FOLDER = "/home/qai/workplace/ord-data/data"


def ord_prepare_db():
    engine = create_engine(CONNECTING_STRING, future=True)
    prepare_database(engine)


def ord_add_data(dataset_id: str, local_data_folder: str = LOCAL_DATA_FOLDER, uspto_only=True):
    if local_data_folder:
        p = [*glob.glob(f"{local_data_folder}/*/{dataset_id}.pb.gz")][0]
        logger.info(f"found local dataset: {p}")
        dataset = load_message(p, Dataset)
    else:
        dataset = fetch_dataset(dataset_id)
    if uspto_only and "uspto-grants-" not in dataset.name.lower():
        logger.critical(f"skipping as not USPTO dataset: {dataset.name}")
        return
    logger.warning("dataset loaded")
    engine = create_engine(CONNECTING_STRING, future=True)
    with Session(engine) as session:
        add_dataset(dataset, session)
        session.flush()
        session.commit()


def ord_add_rdkit():
    engine = create_engine(CONNECTING_STRING, future=True)
    with Session(engine) as session:
        add_rdkit(session)
        session.commit()


def get_dataset_ids():
    with open("setup_db_dataset_ids_20230328.txt", "r") as f:
        return [l.strip().replace(".pb.gz", "") for l in f.readlines()]


def get_existing_dataset_ids():
    engine = create_engine(CONNECTING_STRING, future=True)
    session = Session(engine)
    query = session.query(_MESSAGE_TO_MAPPER[Dataset])
    df = pd.read_sql(query.statement, engine)
    return df['dataset_id'].tolist()


if __name__ == '__main__':
    # ord_prepare_db()  # only run this once
    existing_dataset_ids = get_existing_dataset_ids()
    for index, i in tqdm(enumerate(get_dataset_ids())):
        base_id = i.split("/")[-1]
        if base_id in existing_dataset_ids:
            logger.info(f"dataset already exists: {i}")
            continue
        logger.warning(f"START ADDING {index}: {i}")
        ord_add_data(base_id, uspto_only=True, local_data_folder=LOCAL_DATA_FOLDER)
        logger.warning(f"FINISH ADDING: {i}")
