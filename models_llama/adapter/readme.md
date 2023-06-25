## Finetune with LLaMA adapter and ORD data

### prepare datasets
1. Unzip the cleaned ORD data from [data_from_pb_no_warning_*.7z](../../ord_data/data_from_pb_no_warning_20230416.7z)
2. run `prepare_instructions.py`, datasets will be written to the [data](data)
folder as `ins_*_train.json` and `ins_*_test.json`.

### finetune
See [my fork](https://github.com/qai222/LLaMA-Adapter) to [LLaMA-Adapter](https://github.com/OpenGVLab/LLaMA-Adapter).