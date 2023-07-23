import pandas as pd

df = pd.read_csv("eval_infer.csv")
n = len(df)

print("valid ord:", sum(df['valid_ord']), "/", n)
print(">> flat compound list in reaction inputs")
print(
    "reactions with absent/excess compounds:", df[df['inputs_compounds_list__n_absent_compounds'] > 0].shape[0],
    "/", df[df['inputs_compounds_list__n_excess_compounds'] > 0].shape[0]
)
print(
    "reactions with changed identifiers:",
    df[df['inputs_compounds_list__n_compounds_identifiers_changed'] > 0].shape[0]
)
print(
    "reactions with changed reaction role:",
    df[df['inputs_compounds_list__n_compounds_reaction_role_changed'] > 0].shape[0]
)
print("reactions with changed amount:", df[df['inputs_compounds_list__n_compounds_amount_changed'] > 0].shape[0])
print("identical:", df[df['inputs_compounds_list__average_deep_distance'] == 0].shape[0])
print(">> list of compound lists in reaction inputs")
print("reactions with misplaced groups:", df[df['inputs_compounds_lol__n_misplaced_groups'] > 0].shape[0])
print(">> conditions")
print("reactions with missing conditions:", df[df['conditions__n_erroneous_condition_types'] > 0].shape[0])

"""
valid ord: 1919 / 2000
>> flat compound list in reaction inputs
reactions with absent/excess compounds: 200 / 68
reactions with changed identifiers: 234
reactions with changed reaction role: 371
reactions with changed amount: 178
identical: 1389
>> list of compound lists in reaction inputs
reactions with misplaced groups: 606
>> conditions
reactions with missing conditions: 247
"""