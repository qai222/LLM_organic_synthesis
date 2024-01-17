from loguru import logger

from ord_diff.io import ModelOutput
from ord_diff.utils import json_load

# load the prompt template used for the dataset
prompt_template = json_load("../models_llama/adapter/USPTO-t900/params.json")['prompt_template']
logger.warning(f"prompt template is:\n{prompt_template}")
prompt_header = prompt_template.split("\n")[0]
response_header = prompt_template.split("\n")[-2]

# load one inference from a specific experiment
record = json_load("../models_llama/adapter/infer-expt_202311060037/ord-511ca00092644ce09748fc8056219618.json")
ref_string = record['output']
raw = record['response']

# load the inference as a ModelOutput object
model_output = ModelOutput.from_raw_alpaca(
    raw=raw,
    ref=ref_string,
    identifier=record['reaction_id'],
    prompt_template=prompt_template,
    prompt_header=prompt_header,
    response_header=response_header,
    instruction=record['instruction'],
)

# use protobuf to parse strings to ORD message
from google.protobuf import json_format
from ord_schema.proto import reaction_pb2

reference_message = json_format.Parse(model_output.ref, reaction_pb2.Reaction())
# this would error out if inferred string (completion) is syntactically incorrect
inferred_message = json_format.Parse(model_output.response, reaction_pb2.Reaction())

# find `Compound` messages in a `Reaction` message
from ord_diff.evaluation import get_compounds

compounds_from_inputs = get_compounds(reference_message, extracted_from="inputs")
compounds_from_outcomes = get_compounds(reference_message, extracted_from="outcomes")

# you can convert any message back to dict/JSON using protobuf
inferred_message_dict = json_format.MessageToDict(inferred_message)
inferred_message_json = json_format.MessageToJson(inferred_message)
