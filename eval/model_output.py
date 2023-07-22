from __future__ import annotations

"""
given the reference json and a string response from the llm, evaluate their differences
"""


class modeloutput:

    def __init__(self, identifier: str, raw: str, prompt: str = None, response: str = None, ref: str = None):
        """
        the full output of a model should be
        `prompt_header` + `prompt` + `response_header` + `response`
        there can be arbitrary number of line breaks between any two of them
        """
        self.identifier = identifier
        self.raw = raw
        self.prompt = prompt
        self.response = response
        self.ref = ref

    @classmethod
    def from_raw_alpaca(
            cls, raw: str, ref: str, identifier: str,
            prompt_template="### procedure:\n{instruction}\n\n### ord-json:\n",
            prompt_header="### procedure:",
            response_header="### ord-json:",
    ):
        """
        create from responses to alpaca-like instruction prompts
        """
        assert prompt_header in raw
        assert response_header in raw
        n_line_breaks = prompt_template.count("\n")
        assert n_line_breaks == 4
        assert raw.count("\n") == n_line_breaks

        try:
            p_header, prompt, _, r_header, response = raw.split("\n")
        except Exception as e:
            raise ValueError(f"invalid text: {raw}")
        assert p_header == prompt_header
        assert r_header == response_header
        model_output = cls(identifier, raw, prompt, response, ref)
        return model_output
