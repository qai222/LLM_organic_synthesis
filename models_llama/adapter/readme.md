## Finetune LLaMA using LLaMA adapter and ORD data

### USPTO datasets
There are


1. Unzip the cleaned USPTO ORD data from [data_from_pb_no_warning_*.7z](../../ord_data/data_from_pb_no_warning_20230416.7z)
2. Run `prepare_instructions.py`, datasets will be written to the current folder as `OrdAlpaca_*.json`.
    - you may need a tokenizer for this, ex.
      download from https://huggingface.co/nyanko7/LLaMA-7B/resolve/main/tokenizer.model and put it in a folder named `7B_tokenizer`

### finetune
See [my fork](https://github.com/qai222/LLaMA-Adapter) to [LLaMA-Adapter](https://github.com/OpenGVLab/LLaMA-Adapter).