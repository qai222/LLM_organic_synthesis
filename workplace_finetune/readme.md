## Finetune LLaMA using LLaMA adapter and ORD data

[//]: # (### USPTO datasets)

[//]: # (USPTO datasets used for finetuning/testing are zipped into two files:)

[//]: # (- [USPTO-t900.tar.xz]&#40;USPTO-t900.tar.xz&#41;)

[//]: # (- [USPTO-t1200.tar.xz]&#40;USPTO-t1200.tar.xz&#41;)

[//]: # ()
[//]: # (The integer after "t" represents the max num of tokens. To recreate them:)

[//]: # (1. Unzip the cleaned USPTO ORD data from [data_from_pb_no_warning_*.7z]&#40;../../ord_data/data_from_pb_no_warning_20230416.7z&#41;)

[//]: # (2. Run `prepare_instructions.py`)

[//]: # (    - you may need a tokenizer for this, ex.)

[//]: # (      download from https://huggingface.co/nyanko7/LLaMA-7B/resolve/main/tokenizer.model and put it in a folder named `7B_tokenizer`)

[//]: # ()
[//]: # (### Finetune procedure)

[//]: # (See [my fork]&#40;https://github.com/qai222/LLaMA-Adapter&#41; to [LLaMA-Adapter]&#40;https://github.com/OpenGVLab/LLaMA-Adapter&#41;.)

[//]: # ()
[//]: # (### Raw infer results)

[//]: # (- [infer-expt_202311060037.7z]&#40;infer-expt_202311060037.7z&#41; LLaMA infer results from the test set of [USPTO-t900.tar.xz]&#40;USPTO-t900.tar.xz&#41;.)

[//]: # (- [infer-expt_202311062152.7z]&#40;infer-expt_202311062152.7z&#41; LLaMA-2 infer results from the test set of [USPTO-t900.tar.xz]&#40;USPTO-t900.tar.xz&#41;.)

[//]: # ()
