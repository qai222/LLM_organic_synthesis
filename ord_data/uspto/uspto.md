## USPTO DATA

This folder contains USPTO data extracted from ORD. 
The scripts for extraction and deduplication are also included.

#### workflow

1. download ORD datasets from https://github.com/open-reaction-database/ord-data
2. use [setup_db.py](setup_db.py) to set up ORD postgres database, a list of dataset ids ([setup_db_dataset_ids_20230328.txt](setup_db_dataset_ids_20230328.txt)) is required
3. use [export_from_pb.py](export_from_pb.py) to extract reactions, 
if at least one warning is raised in extraction, this reaction record is excluded.
All warnings are recorded in the log file.
4. use [dedup.py](dedup.py) to deduplicate the reaction records based on `notes__procedureDetails`, 
this is done through `openai tools fine_tunes.prepare_data`

#### JSON data

An extracted reaction has the following fields:
- `notes__procedureDetails`: raw text describing the reaction procedure
- `inputs`: `ord.Reaction.inputs`
- `conditions`: `ord.Reaction.conditions`
- `workups`: `ord.Reaction.workups`
- `outcomes`: `ord.Reaction.outcomes`
- `warning_messages`: a list of warning messages from extraction, should be empty

the latest json data file is
[data_from_pb_no_warning_20230416_dedup.7z](data_from_pb_no_warning_20230416_dedup.7z)

