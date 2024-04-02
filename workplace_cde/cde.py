import gzip
import json
from typing import List

from chemdataextractor.doc import Paragraph, Sentence
from pandas._typing import FilePath
from tqdm import tqdm

"""
1. CDE requires `python3.8`, best to start with a new venv.
2. CDE auto download bert models when first run
"""


def json_dump(filename, obj, gz=False, indent=None):
    if gz:
        open_file = gzip.open
        assert filename.endswith(".gz")
        with open_file(filename, 'wt', encoding="UTF-8") as f:
            json.dump(obj, f, indent=indent)
    else:
        open_file = open
        with open_file(filename, 'w', encoding="UTF-8") as f:
            json.dump(obj, f, indent=indent)


def json_load(filename):
    if filename.endswith(".gz"):
        open_file = gzip.open
        with open_file(filename, 'rt', encoding="UTF-8") as f:
            return json.load(f)
    else:
        open_file = open
        with open_file(filename, 'r', encoding="UTF-8") as f:
            return json.load(f)


def cde_cner(procedure_text: str) -> List[str]:
    paragraph = Paragraph(procedure_text)
    cems = []
    for s in paragraph.sentences:
        s: Sentence
        cems += [c.text for c in s.cems]
    return cems


def export_cde_cne(dataset_folder: FilePath):
    test_data = json_load(f"{dataset_folder}/test.json")
    test_ids = json_load(f"{dataset_folder}/meta.json")['test_data_identifiers']
    cde_data = dict()
    for tid, data in tqdm(zip(test_ids, test_data)):
        reaction_id = tid
        instruction = data['instruction'].split("###")[1].strip()
        paragraph = Paragraph(instruction)
        cems = []
        for s in paragraph.sentences:
            s: Sentence
            cems += [c.text for c in s.cems]
        cde_data[reaction_id] = cems
    return cde_data


if __name__ == '__main__':
    json_dump(
        "cde.json",
        export_cde_cne(
            dataset_folder="../workplace_data/datasets/USPTO-n100k-t2048_exp1"),
        gz=False, indent=2
    )

# class OrdCompoundModel(QuantityModel):
#     specifier = StringType(parse_expression=I('Tc'), required=True)  # I cannot find a specifier for the amount...
#     compound = ModelType(Compound, required=True)
#     parsers = [QuantityModelTemplateParser(), MultiQuantityModelTemplateParser()]
