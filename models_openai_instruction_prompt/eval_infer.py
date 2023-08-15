from evaluator import *

logger.remove(0)

with open("ord_dataset-03ba810b7f464a06b5d8787af2e8b64e.json_GPT3_5_example0_599.json", "r") as f:
    data = json.load(f)

model_outputs = []
report_summary = EvaluatorReport()
n_invalid_json = 0
n_invalid_ord = 0
for i, record in enumerate(data):
    ref_inputs = record['inputs']
    ref_reaction = {"inputs": ref_inputs, "reaction_id": str(i)}
    ref_string = json.dumps(ref_reaction)
    try:
        act_inputs = json.loads(record['GPT_output_processed'])
        act_reaction = {"inputs": act_inputs, "reaction_id": str(i)}
        act_string = json.dumps(act_reaction)
    except json.decoder.JSONDecodeError:
        # print("invalid JSON", i)
        n_invalid_json += 1
        continue

    model_output = ModelOutput(identifier=str(i), raw="", prompt="", response=act_string, ref=ref_string)
    try:
        e = Evaluator(
            model_output,
            # skip_rule=FieldSkipRule.ignore_absent_in_prompt_with_exceptions,
            skip_rule=FieldSkipRule.ignore_absent_in_prompt,
        )
    except (json_format.ParseError, EvaluatorError) as e:
        # print("invalid ORD", i)
        n_invalid_ord += 1
        continue
    report = e.run_evaluate()
    report_summary += report

print(f"invalid JSON #: {n_invalid_json}/{len(data)}")
print(f"invalid ORD #: {n_invalid_ord}/{len(data)}")

df = report_summary.get_table_compound_fields(from_inputs=True)
df.to_csv("eval__compound_inputs.csv")
df = report_summary.get_table_compound_fields(from_inputs=False)
df.to_csv("eval__compound_outcomes.csv")
df = report_summary.get_table_messages()
df.to_csv("eval__messages.csv")
