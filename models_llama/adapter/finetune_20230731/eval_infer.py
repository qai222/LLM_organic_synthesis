from __future__ import annotations
from evaluator import *


def run_eval(filename: FilePath, slice_indices: tuple[int, int] | None,
             skip_rule=FieldSkipRule.ignore_absent_in_prompt_with_exceptions):
    res = Evaluator.evaluate_llama_inference_json(filename, slice_indices=slice_indices,
                                                  skip_rule=skip_rule)
    report_summary = res["report_summary"]
    df = report_summary.get_table_compound_fields(from_inputs=True)
    df.to_csv("eval__compound_inputs.csv")
    df = report_summary.get_table_compound_fields(from_inputs=False)
    df.to_csv("eval__compound_outcomes.csv")
    df = report_summary.get_table_messages()
    df.to_csv("eval__messages.csv")


if __name__ == '__main__':
    logger.remove(0)
    logger.add(__file__.replace(".py", ".log"))
    run_eval(
        filename="infer-14.json",
        slice_indices=None,
        # skip_rule=FieldSkipRule.ignore_absent_in_prompt_with_exceptions,
        skip_rule = FieldSkipRule.ignore_absent_in_prompt,
    )
