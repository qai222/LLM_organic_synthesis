import json
from evaluator.schema import ModelOutput, Evaluator, EvaluatorSniffError, evaluate_model_outputs

with open("ord_dataset-03ba810b7f464a06b5d8787af2e8b64e.json_GPT3_5_example0_599.json", "r") as f:
    data = json.load(f)

model_outputs = []
for i, record in enumerate(data):
    ref_inputs = record['inputs']
    ref_reaction = {"inputs": ref_inputs, "reaction_id": str(i)}
    ref_string = json.dumps(ref_reaction)
    try:
        act_inputs = json.loads(record['GPT_output_processed'])
        act_reaction = {"inputs": act_inputs, "reaction_id": str(i)}
        act_string = json.dumps(act_reaction)
    except json.decoder.JSONDecodeError:
        print("invalid JSON", i)
        act_string = record['GPT_output_processed']
    model_output = ModelOutput(identifier=str(i), raw="", prompt=None, response=act_string, ref=ref_string)
    model_outputs.append(model_output)
df = evaluate_model_outputs(model_outputs, report_conditions=False)
df.to_csv("eval_infer.csv", index=False)