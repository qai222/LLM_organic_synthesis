import json

import pytest
from tqdm import tqdm

from evaluator.schema import ModelOutput, Evaluator, EvaluatorSniffError


class TestEvaluator:

    @pytest.fixture
    def inference_data_20230716_99(self) -> list[Evaluator]:
        with open("../models_llama/adapter/infer_20230716-99.json", "r") as f:
            data = json.load(f)
        evaluators = []
        for record in data:
            model_output = ModelOutput.from_raw_alpaca(
                raw=record['response'],
                ref=record['output'],
                identifier=record['reaction_id']
            )
            try:
                evaluators.append(Evaluator(model_output))
            except EvaluatorSniffError:
                continue
        return evaluators

    def test_compound_list(self, inference_data_20230716_99):
        for ev in tqdm(inference_data_20230716_99):
            ev.evaluate_inputs_compounds_list()

    def test_compound_lol(self, inference_data_20230716_99):
        for ev in tqdm(inference_data_20230716_99):
            ev.evaluate_inputs_compounds_lol()

    def test_conditions(self, inference_data_20230716_99):
        for ev in tqdm(inference_data_20230716_99):
            ev.evaluate_conditions()
