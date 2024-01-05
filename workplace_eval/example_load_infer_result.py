import pprint

from ord_diff.io import ModelOutput
from ord_diff.utils import json_load
from loguru import logger

prompt_template = json_load("../models_llama/adapter/USPTO-t900/params.json")['prompt_template']
logger.warning(f"prompt template is:\n{prompt_template}")

prompt_header = prompt_template.split("\n")[0]
response_header = prompt_template.split("\n")[-2]

record = json_load("example_infer_result.json")
ref_string = record['output']
raw = record['response']
model_output = ModelOutput.from_raw_alpaca(
    raw=raw,
    ref=ref_string,
    identifier=record['reaction_id'],
    prompt_template=prompt_template,
    prompt_header=prompt_header,
    response_header=response_header,
    instruction=record['instruction'],
)

pprint.pp(model_output.as_dict())