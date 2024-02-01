import chemdataextractor
import json

with open('../models_llama/adapter/finetune_20230731/infer-14.json', 'rt') as f:
    data = json.load(f)

chem_data_extracted = []
for i, entry in enumerate(data):
    if i % 10 == 0:
        print(f'At entry {i}')
    reaction_id = entry['reaction_id']
    instruction = entry['instruction']


    chemdata_output = chemdataextractor.Document(chemdataextractor.doc.Paragraph(instruction)).records.serialize()


    output = {'reaction_id':reaction_id,
              'instruction':instruction,
              'chemdataextractor_output':chemdata_output}
    
    chem_data_extracted.append(output)


with open('chemdataextractor_output.jsonl', 'wt') as f:
    for entry in chem_data_extracted:
        f.write(json.dumps(entry)+ '\n')
