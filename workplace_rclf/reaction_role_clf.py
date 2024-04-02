import os.path
import random
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from ord.utils import json_load

sns.set_theme()
from ord.evaluation.evaluate_functions import get_pair_evaluators, \
    evaluation_document_level, FilePath, \
    evaluation_reaction_role

_THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))


def reaction_role_clf(inference_folder: FilePath, data_folder: FilePath, wdir: FilePath, cot: bool, cre: bool):
    """
    main evaluation function

    :param inference_folder:
    :param data_folder:
    :param wdir:
    :param cot:
    :return:
    """

    pairs = get_pair_evaluators(inference_folder, data_folder, cot, cre)
    repaired_pairs = evaluation_document_level(pairs)
    df = evaluation_reaction_role(repaired_pairs, n_jobs=10)
    df.to_csv("role.csv", index=False)


def plt_confusion_matrix(role_csv: FilePath = "role.csv", mode="all", panel_label="A"):
    f, ax = plt.subplots(figsize=(3, 4))  # one-column
    df = pd.read_csv(role_csv)
    df['inference_name_clean'] = df['inference_name'].apply(
        lambda x: x.lower().replace("-", "").replace(".", "").replace(" ", ""))
    df['reference_name_clean'] = df['reference_name'].apply(
        lambda x: x.lower().replace("-", "").replace(".", "").replace(" ", ""))
    condition_inf_wrong_name = (df['inference_name_clean'] != df['reference_name_clean']) & (
                df['inference_name'] != "COMPOUND_MISSING")
    df.loc[condition_inf_wrong_name, 'inference_role'] = "WRONG_NAME"

    if mode == "multirole":
        # add condition to focus on multirole compounds
        name_to_role = defaultdict(set)
        for n, role in zip(df['reference_name'], df['reference_role']):
            name_to_role[n].add(role)
        multirole_names = [n for n, roles in name_to_role.items() if
                           len(roles) > 1]  # reference_role can only be 1 of 3, no need to check other 2 labels
        df = df.loc[df['reference_name'].isin(multirole_names)]

    elif mode == "baseline":
        name_to_role_baseline = {}
        name_to_role_train = json_load("name_to_roles_baseline.json")
        for n, roles in name_to_role_train.items():
            role_counter = Counter(roles)
            baseline_roles = sorted(role_counter.keys(), key=lambda x: role_counter[x])
            baseline_roles = [r for r in baseline_roles if role_counter[r] == role_counter[baseline_roles[-1]]]
            name_to_role_baseline[n] = random.choice(baseline_roles)

        def get_baseline_role(_row):
            _name = _row["inference_name"]
            _original_infer_role = _row["inference_role"]
            if _original_infer_role in ['COMPOUND_MISSING', 'WRONG_NAME']:
                return _original_infer_role
            try:
                return name_to_role_baseline[_name]
            except KeyError:
                return random.choice(['REACTANT', 'SOLVENT', 'CATALYST'])

        # df['inference_role'] = df['reference_name'].apply(lambda x: name_to_role_baseline[x])  # TODO should I use reference name here?
        df['inference_role'] = df.apply(get_baseline_role, axis=1)

    # TODO dry
    elif mode == "multirole_baseline":
        name_to_role = defaultdict(list)
        for n, role in zip(df['reference_name'], df['reference_role']):
            name_to_role[n].append(role)
        multirole_names = [n for n, roles in name_to_role.items() if
                           len(set(roles)) > 1]  # reference_role can only be 1 of 3, no need to check other 2 labels
        df = df.loc[df['reference_name'].isin(multirole_names)]
        name_to_role_baseline = {}
        name_to_role_train = json_load("name_to_roles_baseline.json")
        for n, roles in name_to_role_train.items():
            role_counter = Counter(roles)
            baseline_roles = sorted(role_counter.keys(), key=lambda x: role_counter[x])
            baseline_roles = [r for r in baseline_roles if role_counter[r] == role_counter[baseline_roles[-1]]]
            name_to_role_baseline[n] = random.choice(baseline_roles)

        def get_baseline_role(_row):
            _name = _row["inference_name"]
            _original_infer_role = _row["inference_role"]
            if _original_infer_role in ['COMPOUND_MISSING', 'WRONG_NAME']:
                return _original_infer_role
            try:
                return name_to_role_baseline[_name]
            except KeyError:
                return random.choice(['REACTANT', 'SOLVENT', 'CATALYST'])

        # df['inference_role'] = df['reference_name'].apply(lambda x: name_to_role_baseline[x])  # TODO should I use reference name here?
        df['inference_role'] = df.apply(get_baseline_role, axis=1)

    print(">" * 10, mode)
    clf_report = classification_report(df['reference_role'], df['inference_role'],
                                       labels=['REACTANT', 'SOLVENT', 'CATALYST'], digits=4)
    print(clf_report)
    # cm_labels = df['reference_role'].unique().tolist() + df['inference_role'].unique().tolist()
    # cm_labels = sorted(set(cm_labels))
    cm_labels = ['REACTANT', 'SOLVENT', 'CATALYST', 'COMPOUND_MISSING', 'WRONG_NAME']
    cm_normalized = confusion_matrix(df['reference_role'], df['inference_role'], labels=cm_labels, normalize="true")
    cm_normalized = cm_normalized[:3]
    cm = confusion_matrix(df['reference_role'], df['inference_role'], labels=cm_labels)
    cm = cm[:3]
    df_cm = pd.DataFrame(cm, index=cm_labels[:3], columns=cm_labels).transpose()
    df_cm_normalized = pd.DataFrame(cm_normalized, index=cm_labels[:3], columns=cm_labels).transpose()

    cm_labels_sns = ['REACTANT', 'SOLVENT', 'CATALYST', 'MISSING', 'ERROR']
    annot = df_cm.astype(str) + "\n" + df_cm_normalized.apply(lambda x: 100 * x).round(2).astype(str) + "%"

    cmap = sns.color_palette("light:black", as_cmap=True)
    # hm = sns.heatmap(data=df_cm, annot=True, ax=ax, vmin=0, vmax=1, fmt=".2%", cmap=cmap)
    hm = sns.heatmap(
        data=df_cm_normalized,
        annot=annot, ax=ax, vmin=0, vmax=1,
        cmap=cmap, cbar=False,
        fmt="",
        linewidths=1, linecolor='black'
    )
    # cbar = ax.collections[0].colorbar
    # cbar.set_ticks([0, .25, .5, .75, 1])
    # cbar.set_ticklabels(['0', '25%', '50%', '75%', '100%'])

    hm.set_yticklabels(cm_labels_sns, rotation=90, fontdict=dict(fontsize=8))
    hm.set_xticklabels(hm.get_xticklabels(), fontdict=dict(fontsize=8))
    ax.invert_yaxis()
    # ax.set_ylabel("Predicted Reaction Role", fontdict=dict(weight='bold'))
    # ax.set_xlabel(f"Recorded Reaction Role", fontdict=dict(weight='bold'))
    # ax.set_ylabel(f"Inferred Reaction Role", fontdict=dict(weight='bold'))
    # f.patch.set_linewidth(10)
    # f.patch.set_edgecolor('pink')
    f.tight_layout()
    f.savefig(f"cm_{mode}.png", dpi=600)
    f.savefig(f"cm_{mode}.eps", dpi=600)


if __name__ == '__main__':
    reaction_role_clf(
        inference_folder="expt_202403020036/epoch14",
        data_folder="../workplace_data/datasets/USPTO-n100k-t2048_exp1",
        wdir="expt_202403020036_epoch14",
        cot=False,
        cre=False,
    )

    plt_confusion_matrix(panel_label="A")
    plt_confusion_matrix(mode="baseline", panel_label="B")
    plt_confusion_matrix(mode="multirole", panel_label="C")
    plt_confusion_matrix(mode="multirole_baseline", panel_label="D")
