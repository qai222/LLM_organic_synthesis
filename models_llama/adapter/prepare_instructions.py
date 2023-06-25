import glob
import json
import random
from typing import Any

import tqdm

ORD_DATASET_FILES = sorted(glob.glob("data_from_pb_no_warning_20230416/data_from_pb_no_warning/*.json"))
ORD_PROCEDURE_FIELD = "notes__procedureDetails"
STRUCTURED_DATA_FIELD = "inputs"
ALPACA_INSTRUCTION = "Extract a JSON of reaction inputs using ORD data schema"
TOO_LONG = 3000  # if a reaction's text length is larger than this, it is excluded


def reaction_to_alpaca(r: dict[str, Any]):
    """
    alpaca training data looks like the following
    {
        "instruction": <a string sentence about what should be done given the `input`>,
        "input": <string>,
        "output": <string>,
    }
    """
    d = {
        "reaction_id": r['reaction_id'],
        "instruction": ALPACA_INSTRUCTION,
        "input": r[ORD_PROCEDURE_FIELD],
        "output": str(r[STRUCTURED_DATA_FIELD]),
    }
    return d


def collect_reactions():
    reactions = []
    for jp in tqdm.tqdm(ORD_DATASET_FILES):
        with open(jp, "r") as f:
            batch_reactions = json.load(f)
            reactions += batch_reactions
    return reactions


def create_datasets(word_length=500, dataset_size=1000, train_test_ratio=0.2, seed=42, show_hist=False):
    all_reactions = collect_reactions()

    reactions = []
    lengths = []
    n_too_long = 0
    for r in all_reactions:
        le = len(str(r[STRUCTURED_DATA_FIELD])) + len(str(r[ORD_PROCEDURE_FIELD]))
        if le > TOO_LONG:
            n_too_long += 1
            continue
        lengths.append(le)
        if le <= word_length:
            reactions.append(r)
    lengths = sorted(lengths)
    print("criterion too long:", TOO_LONG)
    print("criterion pool:", word_length)
    print("n_all_reactions:", len(all_reactions))
    print("n_too_long:", n_too_long)
    print("n_pool:", len(reactions), "{:.3f}".format(len(reactions) / (len(all_reactions) - n_too_long)))

    if show_hist:
        import plotly.express as px
        import plotly
        fig = px.ecdf(
            x=lengths,
            marginal="histogram",
            labels={
                "x": "prompt + response length",
            }
        )
        plotly.offline.plot(fig, filename='ecdf.html')

    random.seed(seed)
    reactions = random.sample(reactions, int(dataset_size * (1 + train_test_ratio)))
    random.shuffle(reactions)
    alpaca_data = [reaction_to_alpaca(r) for r in reactions]
    train, test = alpaca_data[:dataset_size], alpaca_data[dataset_size:]
    return train, test


def save_datasets(word_length=500, dataset_size=1000, train_test_ratio=0.2, seed=42,
                  name="alpaca_data", show_hist=False):
    train, test = create_datasets(word_length, dataset_size, train_test_ratio, seed, show_hist)

    with open(f"data/{name}_train.json", "w") as f:
        json.dump(train, f)

    with open(f"data/{name}_test.json", "w") as f:
        json.dump(test, f)


if __name__ == '__main__':
    save_datasets(
        word_length=900,
        dataset_size=500,
        train_test_ratio=0.2,
        seed=42,
        name="ins_900",
        # show_hist=False,
        show_hist=True,
    )

"""
criterion too long: 3000
criterion pool: 900
n_all_reactions: 190764
n_too_long: 504
n_pool: 60822 0.320
"""
