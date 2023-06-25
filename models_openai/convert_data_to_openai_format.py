import json

def convert_entries(entries):
    converted_entries = []

    for entry in entries:
        completion_dict = {
            "output_reaction_inputs": entry["output_reaction_inputs"],
            "output_reaction_conditions": entry["output_reaction_conditions"]
        }
        completion_str = json.dumps(completion_dict)
        
        # At inference time, if all goes well, a dictionary can be recovered from the model output as follows:
        # completion_dict = json.loads(completion_str)

        new_entry = {
            "prompt": entry["input_text"] + "\n\n###\n\n",
            "completion": " " + completion_str + "###"
        }
        converted_entries.append(new_entry)

    return converted_entries

for filename in {"data10_v2", "data1000_v2"}:
    # read the data from the json file
    with open(filename + ".json", "r") as f:
        data = json.load(f)

    # convert the data to the format required by openai
    converted_data = convert_entries(data)

    # save to json file
    with open(filename + "_openai.json", "w") as f:
        json.dump(converted_data, f)
