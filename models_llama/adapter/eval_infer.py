import json

from evaluator.schema import ModelOutput, evaluate_model_outputs

if __name__ == '__main__':

    with open("infer_20230716-99.json", "r") as f:
        data = json.load(f)

    model_outputs = []
    for record in data:
        model_output = ModelOutput.from_raw_alpaca(
            raw=record['response'],
            ref=record['output'],
            identifier=record['reaction_id']
        )
        model_outputs.append(model_output)

    df = evaluate_model_outputs(model_outputs)
    df.to_csv("eval_infer.csv", index=False)
