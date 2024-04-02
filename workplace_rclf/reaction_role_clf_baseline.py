import json
from collections import defaultdict

from ord.utils import json_load, json_dump

train_data = json_load("../workplace_data/datasets/USPTO-n100k-t2048_exp1/train.json")
name_to_roles = defaultdict(list)
for d in train_data:
    ref = json.loads(d['output'])
    for r_input in ref['inputs'].values():
        for c in r_input['components']:
            name = c['identifiers'][0]['value']
            role = c['reaction_role']
            name_to_roles[name].append(role)
json_dump("name_to_roles_baseline.json", name_to_roles)
