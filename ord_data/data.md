### LLM data

An extracted reaction has the following fields:

- `notes__procedureDetails`: raw text describing the reaction procedure
- `inputs`: `ord.Reaction.inputs`
- `conditions`: `ord.Reaction.conditions`
- `workups`: `ord.Reaction.workups`
- `outcomes`: `ord.Reaction.outcomes`
- `warning_messages`: a list of warning messages from extraction, should be empty if used in training/testing

### USPTO

The folder [uspto](uspto) contains USPTO data extracted from ORD.
The latest json data file is
[data_from_pb_no_warning_20230416_dedup.7z](uspto/data_from_pb_no_warning_20230416_dedup.7z)

The scripts for extraction and deduplication are also included, to recreate:

1. download ORD datasets from https://github.com/open-reaction-database/ord-data
2. use [setup_db.py](uspto/setup_db.py) to set up ORD postgres database, a list of dataset
   ids ([setup_db_dataset_ids_20230328.txt](uspto/setup_db_dataset_ids_20230328.txt)) is required
3. use [export_from_pb.py](uspto/export_from_pb.py) to extract reactions,
   if at least one warning is raised in extraction, this reaction record is excluded.
   All warnings are recorded in the log file.
4. use [dedup.py](uspto/dedup.py) to deduplicate the reaction records based on `notes__procedureDetails`,
   this is done through `openai tools fine_tunes.prepare_data`

### ChemRxnExtractor

The folder [cre](cre) contains data extracted
from [ChemRxnExtractor](https://github.com/jiangfeng1124/ChemRxnExtractor/).
The latest json data file is [CRE_data.json](cre/CRE_data.json)

The scripts for extraction are also included:

1. download data use the script [download.sh](cre/download.sh), this gives the txt files needed
2. run [export_ord.py](cre/export_ord.py) to export JSON file
