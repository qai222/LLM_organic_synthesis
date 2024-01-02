class ModelOuputError(Exception):
    pass


class ModelOutput:

    def __init__(self, identifier: str, raw: str, prompt: str = None, response: str = None, ref: str = None,
                 instruction: str = None):
        """
        the full output of a model should be
        `prompt_header` + `prompt` + `response_header` + `response`
        there can be arbitrary number of line breaks between any two of them
        """
        self.instruction = instruction
        self.identifier = identifier
        self.raw = raw
        self.prompt = prompt
        self.response = response
        self.ref = ref

    def as_dict(self):
        return {
            "identifier": self.identifier,
            "prompt": self.prompt,
            "response": self.response,
            "reference": self.ref
        }

    @staticmethod
    def parse_raw(
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
            prompt, response = ModelOutput.parse_raw(raw, prompt_template, prompt_header, response_header)
        except Exception:
            raise ModelOuputError
        model_output = cls(identifier, raw, prompt, response, ref, instruction)
        return model_output
