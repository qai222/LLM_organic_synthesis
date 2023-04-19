# Extracting Structured Data from Free-form Organic Synthesis Text

This project consists of the following parts (branches):
- [openai-model](https://github.com/qai222/LLM_organic_synthesis/tree/openai-model): for fined-tuned models using OPENAI
- [opensource-model](https://github.com/qai222/LLM_organic_synthesis/tree/opensource-model): for open source models
- [dash-app](https://github.com/qai222/LLM_organic_synthesis/tree/dash_app): demo app for visualizing model inferences
- [ord-data](https://github.com/qai222/LLM_organic_synthesis/tree/ord_data): data entries from [the open reaction database](https://open-reaction-database.org/) for fine tuning


## Hackathon demo
A [demo](https://qai222.github.io/LLM_organic_synthesis/) hosted using Github pages is available.

This model is an OPENAI Davinci model fine-tuned with [300 ORD data entries](https://github.com/qai222/LLM_organic_synthesis/blob/dash_app/dash_app/assets/data1000_v2.json)
and evaluated using another set of [50 ORD data entries](https://github.com/qai222/LLM_organic_synthesis/blob/dash_app/dash_app/assets/data50_v2_test.json)
