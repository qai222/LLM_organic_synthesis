from __future__ import annotations

from pydantic import BaseModel


class ModelOuputError(Exception):
    pass


class ModelOutput(BaseModel):
    """
    the full output of a model should be
    `prompt_header` + `prompt` + `response_header` + `response`
    there can be arbitrary number of line breaks between any two of them
    """
    identifier: str
    raw: str
    prompt: str | None = None
    response: str | None = None
    ref: str | None = None
    instruction: str | None = None

    def as_dict(self):
        return {
            "identifier": self.identifier,
            "prompt": self.prompt,
            "response": self.response,
            "reference": self.ref
        }

    @staticmethod
    def parse_raw_output(
            raw,
            prompt_template="### Procedure:\n{instruction}\n\n### ORD-JSON:\n",
            prompt_header="### Procedure:",
            response_header="### ORD-JSON:",
    ):
        assert prompt_header in raw
        assert response_header in raw
        n_line_breaks = prompt_template.count("\n")
        assert n_line_breaks == 4
        assert raw.count("\n") == n_line_breaks

        p_header, prompt, _, r_header, response = raw.split("\n")
        # try:
        #     p_header, prompt, _, r_header, response = raw.split("\n")
        # except Exception as e:
        #     raise ModelOuputError(f"invalid text: {raw}")
        assert p_header == prompt_header
        assert r_header == response_header
        return prompt, response

    @classmethod
    def from_raw_alpaca(
            cls, raw: str, ref: str, identifier: str,
            prompt_template="### Procedure:\n{instruction}\n\n### ORD-JSON:\n",
            prompt_header="### Procedure:",
            response_header="### ORD-JSON:",
            instruction: str = None,
    ):
        """ create from responses to alpaca-like instruction prompts """
        try:
            prompt, response = ModelOutput.parse_raw_output(raw, prompt_template, prompt_header, response_header)
        except Exception:
            raise ModelOuputError
        model_output = cls(
            identifier=identifier, raw=raw, prompt=prompt, response=response, ref=ref, instruction=instruction
        )
        return model_output
