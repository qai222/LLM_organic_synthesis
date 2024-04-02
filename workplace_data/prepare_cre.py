import pathlib

from pandas._typing import FilePath

from ord.data.data_llm import DEFAULT_PROMPT_TEMPLATE, ReactionData, LlmData
from ord.utils import json_dump, json_load


def make_cre_dataset_singular(cre_singular_json: FilePath):
    data = json_load(cre_singular_json)
    records = []
    meta_record = dict(test_data_identifiers=[], prompt_template=DEFAULT_PROMPT_TEMPLATE)
    for d in data:
        reaction_data = ReactionData(**d)
        llm_data = LlmData.from_reaction_data(
            reaction_data, ['inputs', 'conditions', 'workups', 'outcomes'],
            DEFAULT_PROMPT_TEMPLATE
        )
        record = {
            "instruction": DEFAULT_PROMPT_TEMPLATE.format(reaction_data.procedure_text),
            "output": llm_data.reference_completion
        }
        records.append(record)
        meta_record['test_data_identifiers'].append(reaction_data.reaction_id)
    pathlib.Path("datasets/CRE_sinular").mkdir(exist_ok=True)
    json_dump("datasets/CRE_sinular/test.json", records, indent=2)
    json_dump("datasets/CRE_sinular/meta.json", meta_record, indent=2)


if __name__ == '__main__':
    make_cre_dataset_singular("cre/CRE_data_singular.json")
