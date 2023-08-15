import pytest
from tqdm import tqdm

from evaluator import Evaluator, DiffReportKind, OrdMajorField


class TestEvaluator:

    @pytest.fixture
    def evaluators_from_20230716_99(self) -> list[Evaluator]:
        return Evaluator.evaluators_from_json("../models_llama/adapter/finetune_20230716/infer-99.json",
                                              slice_indices=(0, 100))[0]

    def test_diff_report_1(self, evaluators_from_20230716_99):
        for ev in tqdm(evaluators_from_20230716_99):
            ev: Evaluator
            ev.get_diff_report(kind=DiffReportKind.LIST_OF_COMPOUNDS, in_field=OrdMajorField.inputs)

    def test_diff_report_2(self, evaluators_from_20230716_99):
        for ev in tqdm(evaluators_from_20230716_99):
            ev: Evaluator
            ev.get_diff_report(
                kind=DiffReportKind.LIST_OF_COMPOUND_LISTS, in_field=OrdMajorField.inputs
            )

    def test_diff_report_3(self, evaluators_from_20230716_99):
        for ev in tqdm(evaluators_from_20230716_99):
            ev: Evaluator
            ev.get_diff_report(
                kind=DiffReportKind.LIST_OF_COMPOUNDS, in_field=OrdMajorField.outcomes
            )

    def test_diff_report_4(self, evaluators_from_20230716_99):
        for ev in tqdm(evaluators_from_20230716_99):
            ev: Evaluator
            ev.get_diff_report(
                kind=DiffReportKind.LIST_OF_COMPOUND_LISTS, in_field=OrdMajorField.outcomes
            )

    def test_diff_report_5(self, evaluators_from_20230716_99):
        for ev in tqdm(evaluators_from_20230716_99):
            ev: Evaluator
            ev.get_diff_report(kind=DiffReportKind.LIST_OF_REACTION_WORKUPS, in_field=OrdMajorField.workups)

    def test_diff_report_6(self, evaluators_from_20230716_99):
        for ev in tqdm(evaluators_from_20230716_99):
            ev: Evaluator
            ev.get_diff_report(kind=DiffReportKind.REACTION_CONDITIONS, in_field=OrdMajorField.conditions)
