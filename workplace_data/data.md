### USPTO

The processed USPTO data (USPTO-master.json.gz) is available on [figshare]()

The scripts for extraction and deduplication are also included, to recreate:

1. download ORD datasets from https://github.com/open-reaction-database/ord-data
2. use [export_from_pb.py](uspto/export_from_pb.py) to extract and deduplicate reactions, this gives `export_from_pb_dedup.json.gz`
3. use [prepare_uspto](prepare_uspto.py) to make an 8:1:1 split, 
this gives the folder [USPTO-n100k-t2048_exp1](datasets/USPTO-n100k-t2048_exp1), 
which contains train-val-test json files. 
This folder is compressed into [a 7z file](datasets/USPTO-n100k-t2048_exp1.7z).

### ChemRxnExtractor

The folder [cre](cre) contains data extracted
from [ChemRxnExtractor](https://github.com/jiangfeng1124/ChemRxnExtractor/).
We used unireaction dataset [CRE_data_singular.json](cre/CRE_data_singular.json)

The scripts for extraction are also included:
1. download data use the script [download.sh](cre/download.sh), this gives the txt files needed
2. run [export_from_cre.py](cre/export_from_cre.py) to export JSON file.
3. [prepare_cre.py](prepare_cre.py) will make the folder [CRE_singular](datasets/CRE_sinular) 
so it can be plugged into evaluation workflows.
