# Extracting Structured Data from Organic Synthesis Procedures Using a Fine-Tuned Large Language Model

Organic synthesis procedures are traditionally represented by free-form texts.
This project explores how large language models can convert such unstructured
texts to structured data, so they can be used for downstream data science or machine learning
applications.

[//]: # (### Demo page: OPENAI inference)

[//]: # ()
[//]: # (The app in [demo_apps/dash_app]&#40;demo_apps/dash_app&#41; shows inference results from fine-tuned OPENAI models.)

[//]: # (OPENAI API key is required in [the deployment script]&#40;demo_apps/dash_app/test_deploy.sh&#41;.)

## Folders
- [workplace_data](workplace_data): Data downloading and processing. Most of the organic synthesis procedures, free text or structured, came from
[the Open Reaction Database](https://docs.open-reaction-database.org/).
- [workplace_cde](workplace_cde): Comparing with [chemdataextractor2](https://github.com/CambridgeMolecularEngineering/chemdataextractor2/tree/master).
- [workplace_evaluation](workplace_evaluation): Model evaluations.
- [workplace_finetune](workplace_finetune): Finetuning using LLaMA-Adapter.
- [workplace_rclf](workplace_rclf): Reaction role classification.

[//]: # (## About)

[//]: # ()
[//]: # (The current team &#40;04/2024&#41; includes:)

[//]: # ()
[//]: # (- Qianxiang Ai)

[//]: # (- Fanwang Meng)

[//]: # (- Jiale Shi)

[//]: # (- Brenden Pelkie)

[//]: # (- Hassan Harb)

## 2023/03/29 Hackathon
This project was conceived during the
[LLM Hackathon](https://sm.linkedin.com/posts/benblaiszik_llm-march-madness-materials-chemistry-hackathon-activity-7041802764464648192-gjXo)
on 2023/03/29.
We thank Ben Blaiszik for his generous financial support to this project.

For more details, see
- The [2-min demo video](https://twitter.com/QaiAlex/status/1641468953573613569?s=20) by Marcus Schwarting.
- **Section I.C.b** of the preprint [arXiv:2306.06283](https://arxiv.org/pdf/2306.06283.pdf).
- The [demo app](https://qai222.github.io/LLM_organic_synthesis/) on GitHub pages.
  - This app [demo_apps/github_page](demo_apps/github_page) shows precomputed inference results from an `OPENAI davinci`
model. It is a static page from `Dash` using Epix Zhang's [code](https://gist.github.com/exzhawk/33e5dcfc8859e3b6ff4e5269b1ba0ba4?permalink_comment_id=4001137), and is synced to the [github_page](https://github.com/qai222/LLM_organic_synthesis/tree/github_pages) branch.
- Data processing and inference scripts for OPENAI models can be found in the folder [models_openai](models_openai). 
These models are fine-tuned with [300 data points](demo_apps/dash_app/assets/data1000_v2.json)
and evaluated using another set of [50 data points](demo_apps/dash_app/assets/data50_v2_test.json).

