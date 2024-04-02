import glob
import json

from google.protobuf.json_format import ParseDict, ParseError
from json_repair import repair_json
from ord_schema.message_helpers import find_submessages
from ord_schema.proto.reaction_pb2 import Reaction, Compound, ProductCompound

from ord.utils import json_load, json_dump

data = []
reaction_ids = json_load("../../workplace_data/datasets/USPTO-n100k-t2048_exp1/meta.json")['test_data_identifiers']
for jf in sorted(glob.glob("../expt_202403020036/epoch14/*.json")):
    data += json_load(jf)

output = dict()
for reaction_id, d in zip(reaction_ids, data):
    text = repair_json(d)
    try:
        reaction = ParseDict(json.loads(text), Reaction())
    except ParseError:
        output[reaction_id] = []
        continue
    reaction: Reaction
    compounds = find_submessages(reaction, Compound)
    compounds += find_submessages(reaction, ProductCompound)
    chemical_names = []
    for c in compounds:
        try:
            chemical_names.append(c.identifiers[0].value)
        except IndexError:
            continue
    output[reaction_id] = chemical_names

json_dump("llm.json", output, indent=2)
